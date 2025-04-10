'''
@author: James V. Talwar
Created on 2/23/2024 @ 19:55:16

About: YuriTheTrainerWhoTrains.py is a distributed-trainer class for VADEr models for complex disease prediction. Training 
is implemented by MakeItTrain.py. 
'''

import logging
import os
import pandas as pd
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from MetricUtils import BinaryClassAccuracy, MultiClassAccuracy, Calc_ROC_AUC 
from TrainingUtils import MixUp, SupCon, MixAndMatch, SmoothLabels #, RMSNorm


logger = logging.getLogger(__name__)


'''
About: Method to identify last checkpointed model path (by most recent step/epoch) for loading 
and resuming training.

Input: checkpointPath: String corresponding to the directory where model checkpoints exist.
Output: String corresponding the the full path of the latest/last checkpointed model.
'''

def IdentifyLastCheckpoint(checkpointPath):
    savedModels = [modelName for modelName in os.listdir(checkpointPath) if modelName.endswith(".pt")]
    
    if len(savedModels) == 0:
        return None
    
    maxStepsCompleted = -1
    lastModel = None
    
    for modelName in savedModels:
        numTrainingSteps = int(modelName.split("_")[-1].split(".")[0])
        if numTrainingSteps > maxStepsCompleted:
            maxStepsCompleted = numTrainingSteps
            lastModel = modelName
    
    return os.path.join(checkpointPath, lastModel)


'''
About: Method to remove duplicates from gathered predictions/labels for validation data
       when conducting distributed training (and thus ensuring accurate validation metrics)
       unaffected by duplication. Call only if drop_last in DistributedSampler is set to False (default)
Input(s): baseDuplicatedRank: Int corresponding to the first rank where duplications/padding occur 
                             (baseDuplicatedRank == 0 equates to no duplications).
          gatheredList: List corresponding to all gathered tensors from world of GPUs, with each index 
                        corresponding to the GPU rank corresponding tensors.
Output: gatheredList: List with add duplicated/padded elements removed.
'''
def RemovePaddedDuplicates(baseDuplicatedRank, gatheredList):
    if baseDuplicatedRank:
        for i in range(baseDuplicatedRank, len(gatheredList)):
            gatheredList[i] = gatheredList[i][:-1]
    
    return gatheredList


class Trainer: #NOTE: Initialize manual seed for consistency across things
    '''
    Parameters: 
        model: (VADEr) model for which training is to be conducted.
        trainLoader: Distributed Dataloader for the training set. DistributedSampler should be set to True.
        valLoader: Distributed Dataloader for the validation set. DistributedSampler should be set to False.
        optimizer: Optimizer for training and model parameter updates.
        scheduler: Scheduler that updates learning rate as a function of the number of steps taken. 
        checkpointPath: String corresponding to the directory for model checkpointing.
        checkpointFreq: Integer corresponding to the frequency of checkpointing. Used in conjunction with 
                        checkpointFreqMethod to determine whether should occur at an epoch or step resolution.
        trainingSummaryPath: String corresponding to the full path where training and validation metric logs should
                             be written.
        trainingObjective: String in {"mt_cross_entropy", "st_cross_entropy", "sup_con"} corresponding to the desired 
                           training objective to use.
        checkpointFreqMethod: String in {"Epochs", "Steps"} corresponding to whether model checkpointing should
                              be performed at an epoch or step level. Default: 'Epochs'
        gradientClipping: Float corresponding to the value for gradient clipping (by norm) during training. If 0 or None, 
                          no gradient clipping is employed.
        gradientAccumulation: Integer corersponding to the number of gradient accumulation steps to employ for training. 
                              If 0 (default) or None, gradient accumultation is not employed. 
        trainEvalLoader: Distributed Dataloader for the training set. DistributedSampler should be set to False. Needed/Employed 
                         for training set evaluation when checkpointFreqMethod == "Steps". Default = None
    '''
    def __init__(self, 
                 model: nn.Module, 
                 trainLoader: DataLoader, 
                 valLoader: DataLoader, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: Any, 
                 checkpointPath: str, 
                 checkpointFreq: int, 
                 trainingSummaryPath: str,
                 trainingObjective: str,
                 checkpointFreqMethod: str = "Epochs", 
                 gradientClipping: float = 0.0,
                 gradientAccumulation: int = 0,
                 patchDropout: bool = False,
                 mask: torch.tensor = None,
                 trainEvalLoader: DataLoader = None): 
        
        assert checkpointFreqMethod in {"Epochs", "Steps"}, f"checkpoint method must either be training epochs or steps in {'Epochs', 'Steps'}."
        assert trainingObjective in {"mt_cross_entropy", "st_cross_entropy", "sup_con"},  f"training objective {trainingObjective} invalid. Must be one of: 'mt_cross_entropy', 'st_cross_entropy', 'sup_con'."
        if checkpointFreqMethod == "Steps":
            assert isinstance(trainEvalLoader, DataLoader), """trainEvalLoader not specified. Separate (unshuffled) loader is needed for training set evaluation 
            when employing STEP-Based training and checkpointing."""
            self.trainEvalLoader = trainEvalLoader

        self.gpuID = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpuID)
        self.gradientClipping = gradientClipping
        self.mask = mask
        self.enablePretraining = False #Boolean parameter to track supervised contrastive pretraining

        if self.mask is not None:
            self.mask.to(self.gpuID)

        if gradientAccumulation == 1: #gradient accumulation at 1 == no gradient accumulation
            self.gradientAccumulation = 0
        else:
            self.gradientAccumulation = gradientAccumulation

        self.trainLoader = trainLoader
        self.valLoader = valLoader

        #Handle duplications that can occur as a result of padding with DistributedSampler and all samples used (i.e., drop_last = False)
        self.trainDuplicationRank = len(trainLoader.dataset) % torch.distributed.get_world_size() 
        logger.info(f"TRAIN DUPLICATION RANK: {self.trainDuplicationRank}")
        self.valDuplicationRank = len(valLoader.dataset) % torch.distributed.get_world_size() 
        logger.info(f"VALIDATION DUPLICATION RANK: {self.valDuplicationRank}")
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.epochsRun = 0
        self.maxTrainingSteps = None #Total number of training steps to run - specified when train is called 
        
        self.checkpointFreqMethod = checkpointFreqMethod
        self.checkpointFreq = checkpointFreq
        self.checkpointPath = checkpointPath
        self.trainingSummaryPath = trainingSummaryPath
        self.trainingObjective = trainingObjective

        if trainingObjective == "sup_con":
            if self.model.projectionNetwork.bias.shape[0] == 1: #Pretrained model - final layer (projectionNetwork) is now trained with BCE
                logging.info("Setting loss function of pretrained sup con model to BCE for linear classifier training...")
                self.lossFunctions = {"disease": nn.BCEWithLogitsLoss()} 
                
                if self.gpuID == 0: #Only main process needs to keep track of training summary statistics
                    self.trainingStats = pd.DataFrame(columns = ["Train_Disease_Accuracy", "Train_Ancestry_Accuracy", "Train_AUC", "Train_Loss", "Val_Disease_Accuracy", "Val_Ancestry_Accuracy", "Val_AUC", "Val_Loss"])
                
            else: 
                self.lossFunctions = SupCon(device_rank = self.gpuID)
                self.enablePretraining = True
                
                if self.gpuID == 0: 
                    self.trainingStats = pd.DataFrame(columns = ["Train_Loss", "Val_Loss"])
                        

        else:
             if self.gpuID == 0: #Only main process needs to keep track of training summary statistics
                self.trainingStats = pd.DataFrame(columns = ["Train_Disease_Accuracy", "Train_Ancestry_Accuracy", "Train_AUC", "Train_Loss", "Val_Disease_Accuracy", "Val_Ancestry_Accuracy", "Val_AUC", "Val_Loss"])

             if trainingObjective == "mt_cross_entropy":
                 self.lossFunctions = {"disease": nn.BCEWithLogitsLoss(), "ancestry": nn.CrossEntropyLoss()} #can also have one for any other variables of interest - e.g. BCE for disease family history if all individuals had that label (as opposed to a small percentage).
                 
             else: 
                 self.lossFunctions = {"disease": nn.BCEWithLogitsLoss()}

        if self.gpuID == 0: #Only main process needs to keep track of training summary statistics
            self.currentStepTrainingStats = dict()

        assert os.path.isdir(checkpointPath), f"Checkpoint path {checkpointPath} is not a directory. Exiting..."

        fullCheckpointPath = IdentifyLastCheckpoint(checkpointPath)
        if fullCheckpointPath:
            logger.info("Loading snapshot path...")
            self._Load_Checkpoint(fullCheckpointPath)
            
        self.model = DDP(self.model, device_ids = [self.gpuID])

        #Initialize MixUp params - populated (if passed in) when .Train() called:
        self.enable_mixup = False
        self.mixup_alpha = None
        self.mixup_method = None #{None, "mixup", "mix_and_match"}

        #Initialize label smoothing param:
        self.label_smoothing = None
        
        #Initialize patch dropout:
        self.enable_patch_dropout = patchDropout
        if self.enable_patch_dropout:
            logger.info("Enabling patch dropout for Trainer...")


    '''
    About: Method to load the most recent/last checkpointed models. Updates all relevant metrics and necessary
           training parameters and states with checkpointed states.

    Input: fullCheckpointPath: String corresponding to the full checkpointed model path to be loaded.
    '''
    def _Load_Checkpoint(self, fullCheckpointPath):
        loc = f"cuda:{self.gpuID}"
        checkpointDict = torch.load(fullCheckpointPath, map_location = loc)
        self.model.load_state_dict(checkpointDict["modelStateDict"])
        self.optimizer.load_state_dict(checkpointDict["optimizerStateDict"]) 
        self.scheduler.load_state_dict(checkpointDict["schedulerStateDict"])
        self.epochsRun = checkpointDict["epochsRun"]
        
        self.trainingStats = pd.read_csv(self.trainingSummaryPath, sep = "\t", index_col = 0) # Baseline run of untrained model is expected to be added to training statistics

        #get the index of training statistics relevant to the pulled checkpoint
        if self.checkpointFreqMethod == "Epochs":
            if self.epochsRun % self.checkpointFreq != 0:  
                logger.warning(f"_Load_Checkpoint STEPS % checkpointFrequency != 0 . Num Epochs: {self.epochsRun}; checkpointFreq: {self.checkpointFreq}")

            checkpointStepNumber = int(self.epochsRun/self.checkpointFreq) 

        else:
            numberTrainingSteps = self.scheduler.state_dict()["last_epoch"]
            if numberTrainingSteps % self.checkpointFreq != 0:
                logger.warning(f"_Load_Checkpoint STEPS % checkpointFrequency != 0 . Num Steps: {numberTrainingSteps}; checkpointFreq: {self.checkpointFreq}")

            checkpointStepNumber = int(numberTrainingSteps/self.checkpointFreq) 

        self.trainingStats = self.trainingStats[:checkpointStepNumber + 1]

    '''
    About: K-checkpointing. Checkpoint the top-k models by validation performance (AUC, loss, etc.) and update training logs. Should
    only be called by main (rank 0) process.

    Input(s): k: Integer corresponding to the maximum number of models to be locally checkpointed based on the defined metric.
              metric: String (in self.trainingStats.columns) to use as the metric for k checkpointing.
    '''
    def _K_Checkpoint(self, k, metric = "Val_Loss"): 
        assert metric in self.trainingStats.columns, f"Invalid metric. Provided metric {metric} not found in tracked training statistics."

        #Update training summary logs
        trainingStatsToAdd =  list() 
        for column in self.trainingStats.columns:
            trainingStatsToAdd.append(self.currentStepTrainingStats.get(column))

        self.trainingStats.loc[len(self.trainingStats)] = trainingStatsToAdd

        self.trainingStats.to_csv(self.trainingSummaryPath, sep = "\t")

        #Checkpoint models
        if self.checkpointFreqMethod == "Epochs":
            assert self.epochsRun % self.checkpointFreq == 0, f"_K_Checkpoint called at incorrect interval for EPOCHS. Num Epochs: {self.epochsRun}; checkpointFreq: {self.checkpointFreq}"
            checkpointStepNumber = int(self.epochsRun/self.checkpointFreq) 
            logger.info(f"Checkpointing at EPOCH {self.epochsRun}")
        else:
            numberTrainingSteps = self.scheduler.state_dict()["last_epoch"]
            assert numberTrainingSteps % self.checkpointFreq == 0, f"_K_Checkpoint called at incorrect interval for STEPS. Num Steps: {numberTrainingSteps}; checkpointFreq: {self.checkpointFreq}"
            checkpointStepNumber = int(numberTrainingSteps/self.checkpointFreq) #aligns with indexing of training statistics
            logger.info(f"Checkpointing for STEP {numberTrainingSteps} at EPOCH {self.epochsRun}")
        
        saveName = f"VADEr_{self.checkpointFreqMethod[:-1]}_{checkpointStepNumber}.pt"
        savePath = os.path.join(self.checkpointPath, saveName)

        savedModels = [os.path.join(self.checkpointPath, modelName) for modelName in os.listdir(self.checkpointPath) if modelName.endswith(".pt")]

        if len(savedModels) < k:
            torch.save({"modelStateDict": self.model.module.state_dict(),
                        "optimizerStateDict": self.optimizer.state_dict(),
                        "schedulerStateDict": self.scheduler.state_dict(),
                        "epochsRun": self.epochsRun},
                        savePath)

        else:
            #determine maximization/minimization objective
            metricType = metric.split("_")[1]

            #check whether current model is better by metric; Loss is minimization metric
            if metricType == "Loss":
                #From trainingStats get the top K performing models by validation performance
                modelStepstoKeep = self.trainingStats[:-1].nsmallest(k, metric).index.to_list()

                if self.currentStepTrainingStats[metric] < self.trainingStats.loc[modelStepstoKeep[-1], metric]:
                    #save model and delete the worst model in the current k-saved subset 
                    torch.save({"modelStateDict": self.model.module.state_dict(),
                        "optimizerStateDict": self.optimizer.state_dict(),
                        "schedulerStateDict": self.scheduler.state_dict(),
                        "epochsRun": self.epochsRun},
                        savePath)
                    
                    checkpointedModelToDelete = os.path.join(self.checkpointPath, f"VADEr_{self.checkpointFreqMethod[:-1]}_{modelStepstoKeep[-1]}.pt")
                    assert checkpointedModelToDelete in savedModels, f"Identified model to delete {checkpointedModelToDelete} not found in {self.checkpointPath}"
                    os.remove(checkpointedModelToDelete)


            else: #Accuracy and AUC are maximization metrics
                modelStepstoKeep = self.trainingStats[:-1].nlargest(k, metric).index.to_list()

                if self.currentStepTrainingStats[metric] > self.trainingStats.loc[modelStepstoKeep[-1], metric]:
                    torch.save({"modelStateDict": self.model.module.state_dict(),
                        "optimizerStateDict": self.optimizer.state_dict(),
                        "schedulerStateDict": self.scheduler.state_dict(),
                        "epochsRun": self.epochsRun},
                        savePath)
                    
                    checkpointedModelToDelete = os.path.join(self.checkpointPath, f"VADEr_{self.checkpointFreqMethod[:-1]}_{modelStepstoKeep[-1]}.pt")
                    assert checkpointedModelToDelete in savedModels, f"Identified model to delete {checkpointedModelToDelete} not found in {self.checkpointPath}"
                    os.remove(checkpointedModelToDelete)
            
        #reset training stats to add
        self.currentStepTrainingStats = dict()

    '''
    About: Method to calculate loss according to the initialized training objective.

    Inputs: 1) modelOutputs: Dictionary of model outputs.
            2) **kwargs: All necessary labels for loss computation.
    Outputs: loss: torch tensor corresponding to model loss.
    '''
    def _Calc_Loss(self, modelOutputs, **kwargs):
        if self.trainingObjective == "mt_cross_entropy":
            if self.mixup_method == "mixup": #if mixup enabled class probabilities are provided as [B x num_ancestry_classes tensor]
                loss = 0.5 * self.lossFunctions["disease"](modelOutputs["disease"], kwargs["disease"].float()) + 0.5 * self.lossFunctions["ancestry"](modelOutputs["ancestry"], kwargs["ancestry"])
            else:
                loss = 0.5 * self.lossFunctions["disease"](modelOutputs["disease"], kwargs["disease"].float()) + 0.5 * self.lossFunctions["ancestry"](modelOutputs["ancestry"], kwargs["ancestry"].squeeze(1).long())
                
        elif self.trainingObjective == "st_cross_entropy":
            loss = self.lossFunctions["disease"](modelOutputs["disease"], kwargs["disease"].float())
        
        else: 
            if self.enablePretraining: #Supervised contrastive pretraining
                loss = self.lossFunctions(modelOutputs, kwargs["disease"])

            else: 
                loss = self.lossFunctions["disease"](modelOutputs, kwargs["disease"].float()) #["disease"]

        return loss
    
    '''
    About: Method to run training step for given batch.

    Input(s):  1) features: Dictionary of CUDA tensors corresponding to patch features
               2) accumulateGradients: Boolean corresponding as to whether to accumulate gradients, or 
                                       update weights.
               3) accumulationSteps: Integer corresponding to the number of accumulation steps (when performing gradient accumulation).
                                     Differs from self.gradientAccumulation as at end-of-epoch number of steps may be < self.gradientAccumulation.
               4) **kwargs: All necessary labels for loss computation and if age inclusion desired, age features as well.
    Output(s): 1) loss: Float corresponding to batch-mean reduced main-task loss.
               2) output: Model prediction dictionary - needed for non-loss metric evaluation.
    '''
    def _Run_Training_Batch(self, features, accumulateGradients, accumulationSteps, **kwargs):
        if accumulateGradients:
            with self.model.no_sync():
                if self.model.module.includeAge: #including age - need to pass in more than clump dictionary
                    output = self.model(dictOfClumps = features, mask = self.mask, age_batch = kwargs["age_batch"].to(self.gpuID), patch_dropout = self.enable_patch_dropout)
                else:
                    output = self.model(dictOfClumps = features, mask = self.mask, patch_dropout = self.enable_patch_dropout)

                loss = self._Calc_Loss(modelOutputs = output, **kwargs)
                lossValue = loss.item() 
                loss = loss/accumulationSteps #normalize loss by number of gradient accumulation steps or end-of-epoch number of steps

                loss.backward()

        else:
            if self.model.module.includeAge: #including age - need to pass in more than clump dictionary
                output = self.model(dictOfClumps = features, mask = self.mask, age_batch = kwargs["age_batch"].to(self.gpuID), patch_dropout = self.enable_patch_dropout)
            else:
                output = self.model(dictOfClumps = features, mask = self.mask, patch_dropout = self.enable_patch_dropout)

            loss = self._Calc_Loss(modelOutputs = output, **kwargs)
            lossValue = loss.item()
            if self.gradientAccumulation:
                loss = loss/accumulationSteps

            loss.backward()

            if self.gradientClipping:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradientClipping)

            self.optimizer.step()
            '''
            Note if using a scheduler like get_linear_schedule_with_warmup the first learning rate == 0 and can be skipped 
            (i.e., putting scheduler.step() before optimizer.step()). However, in the case of non-zero starting learning rate
            schedulers, follow the convention of calling optimizer.step() before scheduler.step().
            '''
            self.scheduler.step()

            #if self.gpuID == 0:
            #    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2).to('cpu') for p in self.model.parameters() if p.grad is not None]), 2)
            #    logger.info(f"GPU {self.gpuID} Epoch {self.epochsRun} total gradient L2 norm: {total_norm}")

            self.optimizer.zero_grad()
        
        return lossValue, output
    
    def _Train_Epoch(self, epoch, **kwargs):
        self.model.train()
        self.trainLoader.sampler.set_epoch(epoch) # need to set sampler per epoch for train loader
        self.optimizer.zero_grad()

        #For metrics
        epochLoss = 0
        trainingSetPreds = torch.Tensor() 
        trainingSetLabels = torch.Tensor() 

        if "cross_entropy" in self.trainingObjective:
            trainingSetAncestryPreds = torch.Tensor() 
            trainingSetAncestryLabels = torch.Tensor() 

        # Handle gradient accumulation flags when gradient accumulation is not specified/required/desired
        accumulateGradients = False
        accumulationSteps = None
        nextTrainLoaderStep = self.gradientAccumulation

        for i, (patchBatch, diseaseStatusBatch, ancestryBatch, fHBatch, zAgeBatch) in enumerate(self.trainLoader):
            gpuClumpBatch = {k:v.to(self.gpuID) for k,v in patchBatch.items()} #features

            if self.enablePretraining: # Account for genomic "semantic dissimilarity" across ancestries (i.e., for each ancestry split by cases and controls) which can cause undesired hard +/- properties
                diseaseStatusBatch = 2*ancestryBatch + diseaseStatusBatch

            diseaseStatusBatch = diseaseStatusBatch.to(self.gpuID) #labels
            originalLabels = diseaseStatusBatch #copy of original labels - needed for when mixup enabled for non-loss metrics

            if self.label_smoothing:
                diseaseStatusBatch = SmoothLabels(hardTargets = diseaseStatusBatch, alpha = self.label_smoothing)

            if self.trainingObjective == "mt_cross_entropy":
                ancestryBatch = ancestryBatch.to(self.gpuID) 
                originalAncestryLabels = ancestryBatch #copy of original ancestry labels - needed for when mixup enabled for non-loss metrics

            if self.enable_mixup:
                if self.mixup_method == "mix_and_match":
                    gpuClumpBatch, diseaseStatusBatch = MixAndMatch(features = gpuClumpBatch, labels = diseaseStatusBatch, alpha = self.mixup_alpha)
                    
                else:
                    if self.trainingObjective == "mt_cross_entropy":
                        gpuClumpBatch, diseaseStatusBatch, ancestryBatch, zAgeBatch = MixUp(features = gpuClumpBatch, labels = diseaseStatusBatch, alpha = self.mixup_alpha, ancestry = ancestryBatch, age = zAgeBatch.to(self.gpuID))
                    else:
                        gpuClumpBatch, diseaseStatusBatch, _ , zAgeBatch  = MixUp(features = gpuClumpBatch, labels = diseaseStatusBatch, alpha = self.mixup_alpha, age = zAgeBatch.to(self.gpuID))
                #assert not torch.eq(diseaseStatusBatch, originalLabels).all().item(), "POINTER ISSUES!!!" # if originalLabels is changed to continuous then roc_auc_score should raise a continuous format issue
            
            if self.gradientAccumulation:
                endOfEpoch = (i + 1) == len(self.trainLoader)
                accumulateGradients = ((i + 1) % self.gradientAccumulation != 0) #or not endOfEpoch)

                if endOfEpoch:
                    accumulateGradients = False
                
                #If am at the last step (including sub-steps) of an epoch reupdate normalization factor accordingly
                if nextTrainLoaderStep > len(self.trainLoader):
                    accumulationSteps = len(self.trainLoader) % self.gradientAccumulation

                else:
                    accumulationSteps = self.gradientAccumulation
                
                #weights will now being updated - move interval by gradientAccumulation number of steps
                if not accumulateGradients:
                    nextTrainLoaderStep += self.gradientAccumulation

                #logger.info(f"EPOCH {self.epochsRun} index {i + 1}: endOfEpoch: {endOfEpoch}; accumulateGradients: {accumulateGradients}; accumulationSteps {accumulationSteps}")

            labelsAndAge = {"disease": diseaseStatusBatch, "ancestry": ancestryBatch, "age_batch": zAgeBatch}
            lossValue, output = self._Run_Training_Batch(features = gpuClumpBatch, 
                                                         accumulateGradients = accumulateGradients,
                                                         accumulationSteps = accumulationSteps,
                                                         **labelsAndAge)
            
            epochLoss += lossValue * diseaseStatusBatch.shape[0] #losses are mean reduced - renormalize to dataset level for logging/tracking
            
            # Store training set predictions and labels when checkpoint method is epoch based
            if (self.checkpointFreqMethod == "Epochs"): 
                if not self.enablePretraining:
                    # Detach and move to cpu to prevent fragmentation and memory leaks - move back to GPU after full pass
                    if self.trainingObjective == "sup_con":
                        trainingSetPreds = torch.cat([trainingSetPreds, output.detach().to("cpu")], dim = 0)
                    else:
                        trainingSetPreds = torch.cat([trainingSetPreds, output["disease"].detach().to("cpu")], dim = 0) 
                
                trainingSetLabels = torch.cat([trainingSetLabels, originalLabels.detach().to('cpu')], dim  = 0)

                if self.trainingObjective != "sup_con":
                    if "ancestry" in output.keys():
                        trainingSetAncestryPreds = torch.cat([trainingSetAncestryPreds, output["ancestry"].detach().to("cpu")], dim  = 0)
                        trainingSetAncestryLabels = torch.cat([trainingSetAncestryLabels, originalAncestryLabels.detach().to("cpu")], dim  = 0)

            #Step-based checkpointing method evaluate:
            if (self.checkpointFreqMethod == "Steps") and ((self.scheduler.last_epoch % self.checkpointFreq == 0) or (self.scheduler.last_epoch == self.maxTrainingSteps)):                 
                self._Validate(loader = self.trainEvalLoader, duplicationRank = self.trainDuplicationRank, prefix = "Train") 
                self._Validate(loader = self.valLoader, duplicationRank = self.valDuplicationRank)
                self._K_Checkpoint(k = kwargs["k"], metric = kwargs["checkpointMetric"])
                self.model.train()
              
        self.epochsRun += 1 #Training epoch complete so update accordingly
        
        #Training metrics
        if (self.checkpointFreqMethod == "Epochs") and (self.epochsRun % self.checkpointFreq == 0):
            with torch.no_grad():
                # Get the total loss - send to process 0:
                epochLoss = torch.tensor(epochLoss).to(self.gpuID)
                torch.distributed.reduce(epochLoss, dst = 0, op = torch.distributed.ReduceOp.SUM)

                #Move predictions and labels from each process back to GPU and gather on rank 0 GPU for validation metrics
                if not self.enablePretraining:
                    trainingSetPreds = trainingSetPreds.to(self.gpuID)
                    trainingSetLabels = trainingSetLabels.to(self.gpuID)
                    if self.trainingObjective != "sup_con":
                        if ("ancestry" in output.keys()):
                            trainingSetAncestryPreds = trainingSetAncestryPreds.to(self.gpuID)
                            trainingSetAncestryLabels = trainingSetAncestryLabels.to(self.gpuID)

                    if self.gpuID == 0:
                        gatherPredList = [torch.ones_like(trainingSetPreds) for i in range(torch.distributed.get_world_size())]
                        gatherLabelList = [torch.ones_like(trainingSetLabels) for i in range(torch.distributed.get_world_size())]

                        if self.trainingObjective != "sup_con":
                            if ("ancestry" in output.keys()):
                                gatherAncestryPredList = [torch.ones_like(trainingSetAncestryPreds) for i in range(torch.distributed.get_world_size())] 
                                gatherAncestryLabelList = [torch.ones_like(trainingSetAncestryLabels) for i in range(torch.distributed.get_world_size())]

                    else:
                        gatherPredList = None
                        gatherLabelList = None

                        if self.trainingObjective != "sup_con":
                            if ("ancestry" in output.keys()):
                                gatherAncestryPredList = None
                                gatherAncestryLabelList = None 

                    torch.distributed.gather(tensor = trainingSetPreds, gather_list = gatherPredList)
                    torch.distributed.gather(tensor = trainingSetLabels, gather_list = gatherLabelList)

                    if self.trainingObjective != "sup_con":
                        if ("ancestry" in output.keys()):
                            torch.distributed.gather(tensor = trainingSetAncestryPreds, gather_list = gatherAncestryPredList)
                            torch.distributed.gather(tensor = trainingSetAncestryLabels, gather_list = gatherAncestryLabelList)
        
                if self.gpuID == 0:
                    #correct denom for total number samples seen (may exceed dataset size when padding employed)
                    epochLoss =  epochLoss/(len(trainingSetLabels) * torch.distributed.get_world_size()) 
                    self.currentStepTrainingStats["Train_Loss"] = epochLoss.item()

                    logger.info("Train_Loss Updated")

                    if self.enablePretraining:
                        logger.info(f"GPU {self.gpuID} completed computing train metrics...")
                        torch.distributed.barrier()
                        logger.info(f"GPU {self.gpuID} proceeding from validation of train set...")
                        return 
                    
                    #Handle potential duplications of points due to padding by distributed sampler
                    gatherPredList = RemovePaddedDuplicates(baseDuplicatedRank = self.trainDuplicationRank, gatheredList = gatherPredList)
                    gatherLabelList = RemovePaddedDuplicates(baseDuplicatedRank = self.trainDuplicationRank, gatheredList = gatherLabelList)

                    gatheredPreds = torch.cat(gatherPredList, dim = 0) 
                    gatheredLabels = torch.cat(gatherLabelList, dim = 0)

                    assert gatheredPreds.shape[0] == len(self.trainLoader.dataset), "gathered disease predictions dimension[0] does not match the number of elements in training dataset."
                    assert gatheredLabels.shape[0] == len(self.trainLoader.dataset), "gathered disease labels dimension[0] does not match the number of elements in training dataset."

                    trainAccuracy = BinaryClassAccuracy(preds = gatheredPreds, labels = gatheredLabels)
                    trainAUC = Calc_ROC_AUC(preds = torch.sigmoid(gatheredPreds).to("cpu"), labels = gatheredLabels.to("cpu"))
                    
                    self.currentStepTrainingStats["Train_Disease_Accuracy"] = trainAccuracy
                    self.currentStepTrainingStats["Train_AUC"] = trainAUC

                    if self.trainingObjective != "sup_con":
                        if ("ancestry" in output.keys()):
                            gatherAncestryPredList = RemovePaddedDuplicates(baseDuplicatedRank = self.trainDuplicationRank, gatheredList = gatherAncestryPredList)
                            gatherAncestryLabelList = RemovePaddedDuplicates(baseDuplicatedRank = self.trainDuplicationRank, gatheredList = gatherAncestryLabelList)

                            gatheredAncestryPreds = torch.cat(gatherAncestryPredList, dim = 0) 
                            gatheredAncestryLabels = torch.cat(gatherAncestryLabelList, dim = 0)

                            assert gatheredAncestryPreds.shape[0] == len(self.trainLoader.dataset), "gathered ancestry predictions dimension[0] does not match the number of elements in train dataset."
                            assert gatheredAncestryLabels.shape[0] == len(self.trainLoader.dataset), "gathered ancestry labels dimension[0] does not match the number of elements in train dataset."

                            trainAncestryAccuracy = MultiClassAccuracy(preds = gatheredAncestryPreds, labels = gatheredAncestryLabels)

                            self.currentStepTrainingStats["Train_Ancestry_Accuracy"] = trainAncestryAccuracy

                    logger.info(f"GPU {self.gpuID} completed computing train metrics...")
                    torch.distributed.barrier()
                    logger.info(f"GPU {self.gpuID} proceeding from validation of train set...")
                
                else:
                    logger.info(f"GPU {self.gpuID} waiting for train metrics...")
                    torch.distributed.barrier()
                    logger.info(f"GPU {self.gpuID} proceeding from training to validation...")

    

    @torch.no_grad()
    def _Validate(self, loader, duplicationRank, prefix = "Val"):
        self.model.eval()

        validationSetPreds = torch.Tensor() 
        validationSetLabels = torch.Tensor() 

        if "cross_entropy" in self.trainingObjective:
            validationSetAncestryPreds = torch.Tensor() 
            validationSetAncestryLabels = torch.Tensor() 

        epochLoss = 0
        
        for i, (patchBatch, diseaseStatusBatch, ancestryBatch, fHBatch, zAgeBatch) in enumerate(loader):
            gpuClumpBatch = {k:v.to(self.gpuID) for k,v in patchBatch.items()} #features

            if self.enablePretraining: # Account for genomic "semantic dissimilarity" across ancestries (i.e., for each ancestry split by cases and controls) which can cause undesired hard +/- properties
                diseaseStatusBatch = 2*ancestryBatch + diseaseStatusBatch
                
            diseaseStatusBatch = diseaseStatusBatch.to(self.gpuID) #labels

            if self.model.module.includeAge: #including age - need to pass in more than clump dictionary
                output = self.model(dictOfClumps = gpuClumpBatch, mask = self.mask, age_batch = zAgeBatch.to(self.gpuID), patch_dropout = self.enable_patch_dropout)
            else:
                output = self.model(dictOfClumps = gpuClumpBatch, mask = self.mask, patch_dropout = self.enable_patch_dropout)

            if self.trainingObjective == "mt_cross_entropy":
                ancestryBatch = ancestryBatch.to(self.gpuID) 
                loss = 0.5 * self.lossFunctions["disease"](output["disease"], diseaseStatusBatch.float()) + 0.5 * self.lossFunctions["ancestry"](output["ancestry"], ancestryBatch.squeeze(1).long())
                validationSetPreds = torch.cat([validationSetPreds, output["disease"].to("cpu")], dim = 0) 

                validationSetAncestryPreds = torch.cat([validationSetAncestryPreds, output["ancestry"].to("cpu")], dim  = 0) 
                validationSetAncestryLabels = torch.cat([validationSetAncestryLabels, ancestryBatch.to("cpu")], dim  = 0)

            elif self.trainingObjective == "st_cross_entropy":
                ancestryBatch = ancestryBatch.to(self.gpuID)
                loss = self.lossFunctions["disease"](output["disease"], diseaseStatusBatch.float())
                validationSetPreds = torch.cat([validationSetPreds, output["disease"].to("cpu")], dim = 0) 

                if "ancestry" in output.keys():
                    validationSetAncestryPreds = torch.cat([validationSetAncestryPreds, output["ancestry"].to("cpu")], dim  = 0)
                    validationSetAncestryLabels = torch.cat([validationSetAncestryLabels, ancestryBatch.to("cpu")], dim  = 0)

            else:
                loss = self._Calc_Loss(output, disease = diseaseStatusBatch)

                if not self.enablePretraining:
                    validationSetPreds =  torch.cat([validationSetPreds, output.to("cpu")], dim = 0)

            # move to cpu to prevent fragmentation and memory leaks - move back to GPU after full pass; 
            # also used to track total number of datapoints seen per GPU (for instances of dataloader buffering) 
            validationSetLabels = torch.cat([validationSetLabels, diseaseStatusBatch.to('cpu')], dim  = 0)
                
            #losses are mean reduced - renormalize to dataset level for logging/tracking
            epochLoss += loss.item() * diseaseStatusBatch.shape[0] 

        # Get the total loss - send to process 0:
        epochLoss = torch.tensor(epochLoss).to(self.gpuID)
        torch.distributed.reduce(epochLoss, dst = 0, op = torch.distributed.ReduceOp.SUM)

        if not self.enablePretraining:
            #Move predictions and labels from each process back to GPU and gather on rank 0 GPU for validation metrics
            validationSetPreds = validationSetPreds.to(self.gpuID)
            validationSetLabels = validationSetLabels.to(self.gpuID)

            if self.trainingObjective != "sup_con":
                if ("ancestry" in output.keys()):
                    validationSetAncestryPreds = validationSetAncestryPreds.to(self.gpuID)
                    validationSetAncestryLabels = validationSetAncestryLabels.to(self.gpuID)

            if self.gpuID == 0:
                gatherPredList = [torch.ones_like(validationSetPreds) for i in range(torch.distributed.get_world_size())]
                gatherLabelList = [torch.ones_like(validationSetLabels) for i in range(torch.distributed.get_world_size())]

                if self.trainingObjective != "sup_con":
                    if ("ancestry" in output.keys()):
                        gatherAncestryPredList = [torch.ones_like(validationSetAncestryPreds) for i in range(torch.distributed.get_world_size())] 
                        gatherAncestryLabelList = [torch.ones_like(validationSetAncestryLabels) for i in range(torch.distributed.get_world_size())]
            
            else: 
                gatherPredList = None
                gatherLabelList = None

                if self.trainingObjective != "sup_con":
                    if ("ancestry" in output.keys()):
                        gatherAncestryPredList = None
                        gatherAncestryLabelList = None

            torch.distributed.gather(tensor = validationSetPreds, gather_list = gatherPredList)
            torch.distributed.gather(tensor = validationSetLabels, gather_list = gatherLabelList)
        
            if self.trainingObjective != "sup_con":
                if ("ancestry" in output.keys()):
                    torch.distributed.gather(tensor = validationSetAncestryPreds, gather_list = gatherAncestryPredList)
                    torch.distributed.gather(tensor = validationSetAncestryLabels, gather_list = gatherAncestryLabelList)

        if self.gpuID == 0:
            #correct denom for total number samples seen (may exceed dataset size when padding employed)
            epochLoss =  epochLoss/(len(validationSetLabels) * torch.distributed.get_world_size()) 
            self.currentStepTrainingStats[f"{prefix}_Loss"] = epochLoss.item()

            logger.info(f"{prefix}_Loss Updated")

            if self.enablePretraining:
                logger.info(f"GPU {self.gpuID} completed computing {prefix} metrics...")
                torch.distributed.barrier()
                logger.info(f"GPU {self.gpuID} proceeding from validation of {prefix} set...")
        
                return

            #Handle potential duplications of validation points due to padding by distributed sampler
            gatherPredList = RemovePaddedDuplicates(baseDuplicatedRank = duplicationRank, gatheredList = gatherPredList)
            gatherLabelList = RemovePaddedDuplicates(baseDuplicatedRank = duplicationRank, gatheredList = gatherLabelList)

            gatheredPreds = torch.cat(gatherPredList, dim = 0) 
            gatheredLabels = torch.cat(gatherLabelList, dim = 0)

            assert gatheredPreds.shape[0] == len(loader.dataset), f"gathered disease predictions dimension[0] does not match the number of elements in {prefix} dataset."
            assert gatheredLabels.shape[0] == len(loader.dataset), f"gathered disease labels dimension[0] does not match the number of elements in {prefix} dataset."

            validationAccuracy = BinaryClassAccuracy(preds = gatheredPreds, labels = gatheredLabels)
            validationAUC = Calc_ROC_AUC(preds = torch.sigmoid(gatheredPreds).to("cpu"), labels = gatheredLabels.to("cpu"))
            
            self.currentStepTrainingStats[f"{prefix}_Disease_Accuracy"] = validationAccuracy
            self.currentStepTrainingStats[f"{prefix}_AUC"] = validationAUC

            if self.trainingObjective != "sup_con":
                if ("ancestry" in output.keys()):
                    gatherAncestryPredList = RemovePaddedDuplicates(baseDuplicatedRank = duplicationRank, gatheredList = gatherAncestryPredList)
                    gatherAncestryLabelList = RemovePaddedDuplicates(baseDuplicatedRank = duplicationRank, gatheredList = gatherAncestryLabelList)

                    gatheredAncestryPreds = torch.cat(gatherAncestryPredList, dim = 0) 
                    gatheredAncestryLabels = torch.cat(gatherAncestryLabelList, dim = 0)

                    assert gatheredAncestryPreds.shape[0] == len(loader.dataset), "gathered ancestry predictions dimension[0] does not match the number of elements in validation dataset."
                    assert gatheredAncestryLabels.shape[0] == len(loader.dataset), "gathered ancestry labels dimension[0] does not match the number of elements in validation dataset."

                    validationAncestryAccuracy = MultiClassAccuracy(preds = gatheredAncestryPreds, labels = gatheredAncestryLabels)

                    self.currentStepTrainingStats[f"{prefix}_Ancestry_Accuracy"] = validationAncestryAccuracy

            logger.info(f"GPU {self.gpuID} completed computing {prefix} metrics...")
            torch.distributed.barrier()
            logger.info(f"GPU {self.gpuID} proceeding from validation of {prefix} set...")
        
            #logger.info(f"{prefix} current training stats: {self.currentStepTrainingStats}")
        else:
            logger.info(f"GPU {self.gpuID} waiting for validation metrics...")
            torch.distributed.barrier()
            logger.info(f"GPU {self.gpuID} proceeding from validation...")


    '''
    trainingSteps: Number of training steps - only needed when checkpointFreqMethod is step-based. 

    Input(s): mixup_alpha: Float corresponding to the alpha for the beta distribution of the mixup method
    '''
    def Train(self, numberTrainingEpochs, checkpointK, checkpointMetric = "Val_Loss", trainingSteps = None, mixup_alpha = None, mixup_method = None, label_smoothing = None):
        if self.checkpointFreqMethod == "Steps":
            self.maxTrainingSteps = trainingSteps
            
        if mixup_method:
            assert isinstance(mixup_alpha, (int, float)), f"Expecting int/float for mixup corresponding to beta distribution alpha. Provided mixup method {mixup_method} an alpha of {mixup_alpha}"
            assert mixup_alpha > 0, f"Mixup alpha should be positive (> 0 if implementing mixup). Provided mixup: {mixup_alpha} < 0."

            self.enable_mixup = bool(mixup_alpha)
            self.mixup_alpha = mixup_alpha
            self.mixup_method = mixup_method
            
            assert mixup_method in {"mixup", "mix_and_match"}, f"Provided mixup value of {mixup_method} not supported. Supported mixup methods: 'mixup', 'mix_and_match'."
            assert not self.enablePretraining, "Mixup is currently not implemented with supervised contrastive loss pretraining."

            logger.info(f"Mixup enabled to {self.enable_mixup} with an alpha of {self.mixup_alpha} for mixup method {self.mixup_method}")
        
        if label_smoothing:
            assert isinstance(label_smoothing, float), f"Expecting float for label_smoothing alpha. Provided label_smoothing {label_smoothing} is type {type(label_smoothing)}."
            assert label_smoothing > 0 and label_smoothing < 1, f"Alpha for label smoothing should be bounded in (0, 1) range. Provided label smoothing alpha: {label_smoothing}."
            assert not self.enablePretraining, "Label smoothing is currently not implemented with supervised contrastive loss pretraining."

            self.label_smoothing = label_smoothing

            logger.info(f"Label smoothing enabled with an alpha of {self.label_smoothing}.")

        #Baseline (untrained model) evaluate:
        if self.epochsRun == 0:
            self._Validate(loader = self.trainLoader, duplicationRank = self.trainDuplicationRank, prefix = "Train")
            self._Validate(loader = self.valLoader, duplicationRank = self.valDuplicationRank)

            if self.gpuID == 0:
                logger.info(f"GPU {self.gpuID} beginning checkpoint")

                self._K_Checkpoint(k = checkpointK, metric = checkpointMetric)
                
                torch.distributed.barrier()
                logger.info(f"GPU {self.gpuID} proceeding from checkpoint...")

            else:
                logger.info(f"GPU {self.gpuID} waiting for checkpoint completion...")
                torch.distributed.barrier()
                logger.info(f"GPU {self.gpuID} proceeding from checkpoint...")


        for epoch in range(self.epochsRun, numberTrainingEpochs):
            #Train:
            self._Train_Epoch(epoch = epoch, k = checkpointK, checkpointMetric = checkpointMetric)
            
            #Validate and checkpoint if conditions met:
            if (self.checkpointFreqMethod == "Epochs") and ((epoch + 1) % self.checkpointFreq == 0):
                self._Validate(loader = self.valLoader, duplicationRank = self.valDuplicationRank)

                if self.gpuID == 0:
                    logger.info(f"GPU {self.gpuID} beginning checkpoint")

                    self._K_Checkpoint(k = checkpointK, metric = checkpointMetric)
                    
                    torch.distributed.barrier()
                    logger.info(f"GPU {self.gpuID} proceeding from checkpoint...")

                else:
                    logger.info(f"GPU {self.gpuID} waiting for checkpoint completion...")
                    torch.distributed.barrier()
                    logger.info(f"GPU {self.gpuID} proceeding from checkpoint...")


            if self.gpuID == 0:
                logger.info(f"Epoch {self.epochsRun}/{numberTrainingEpochs} complete.")



        
