'''
@author: James V. Talwar
Created on: 3/5/2024 at 07:07:21

About: Train.py performs distributed-training with DDP for VADEr models for complex disease prediction. Inputs expected include a yaml-config path including all necessary details for training and checkpointing 
(e.g., training and dataloading specifications, model hyperparametrs, checkpoint paths etc) and a training objective.

(Single-node multi-gpu) Usage: torchrun --standalone --nproc_per_node=gpu MakeItTrain.py --config CONFIG_PATH 
'''

import os
import logging
import yaml
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from transformers import (get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, 
get_cosine_with_hard_restarts_schedule_with_warmup, get_constant_schedule)

from VADErData import SNP_Dataset
from VADErDataUtils import GenerateChromosomePatchMask
from YuriTheTrainerWhoTrains import Trainer
from vader import VADEr


'''
About: Method to initialize distributed training with DDP.
'''
def DDP_Setup():
    #Increase default timeout to allow for time for full dataset caching
    init_process_group(backend = "nccl", timeout = datetime.timedelta(minutes = 45))
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

'''
About: Load configuration from yaml-config path

Input(s): path: String corresponding to path to yaml config file for training.
'''
def LoadConfig(path):
    return yaml.load(open(path, 'r'), Loader = yaml.SafeLoader)

'''
About: Method to convert training/checkpointing STEPS to corresponding number of epochs for training.

Inputs: 1) maxNumberOfSteps: Integer corresponding to the number of desired training steps.
        2) stepsPerEpoch: Integer corresponding to the length of the trainLoader.
        3) gradientAccumulationSteps: Integer corresponding to desired gradient accumulation number of steps.
                                      Needed to correct effective steps per epoch. Default None. 
Output: epochs: Integer corresponding to the total number of epochs for which to train. 
'''
def StepsToEpochs(maxNumberSteps, stepsPerEpoch, gradientAccumulationSteps = None):
    if gradientAccumulationSteps:
        #Number effective steps is number of perfectly divisible steps + 1 in case of incomplete gradientAccumulationSteps left
        stepsPerEpoch = np.ceil(stepsPerEpoch/gradientAccumulationSteps)

    epochs = np.ceil(maxNumberSteps/stepsPerEpoch) #If ever have a remainder will need another epoch

    return epochs

'''
About: Method to convert number of training EPOCHS to corresponding number of steps. Needed for scheduler 
       max_number_steps and validating scheduler_warmup_steps does not exceed the specified number of 
       total training steps.

Inputs: 1) epochs: Integer corresponding to the specified number of training epochs.
        2) stepsPerEpoch: Integer corresponding to the length of the trainLoader.
        3) gradientAccumulationSteps: Integer corresponding to desired gradient accumulation number of steps.
                                      Needed to correct effective steps per epoch. Default None. 
Output: numSteps: Integer corresponding to the total number of training steps for specified number of epochs.
'''
def EpochsToSteps(epochs, stepsPerEpoch, gradientAccumulationSteps = None):
    if gradientAccumulationSteps:
        #Number effective steps is number of perfectly divisible steps + 1 in case of incomplete gradientAccumulationSteps left 
        stepsPerEpoch = np.ceil(stepsPerEpoch/gradientAccumulationSteps)

    numSteps = stepsPerEpoch*epochs
    
    return numSteps


'''
About: Prepare DDP compatible loader, employing DistributedSampler. Note that with DistributedSampler, shuffle is set to False, and for 
shuffling the loader, sampler is instantiated at each epoch (with set_epoch) to ensure shuffling between epochs.

Input(s): 1) dataset: SNP_Dataset object corresponding to underlying data for loader
          2) shuffle: Boolean corresponding to whether to shuffle the data. In DDP, this is passed to sampler.
Output(s): Dataloader with distributed sampler for training and validating with DDP
'''
def PrepareDistributedLoader(dataset, shuffle, **kwargs):
    loader = DataLoader(dataset, pin_memory = True, shuffle = False, sampler = DistributedSampler(dataset, shuffle = shuffle), **kwargs)
    
    return loader 


'''
About: Method to initialize train and validation dataset objects (i.e., SNP_Dataset objects).

Input: datasetParams: Dictionary of dataset initialization paths and parameters needed to create a SNP_Dataset
                      for both training and validation sets.
Output(s): trainDataset, valDataset: Training and validation SNP_Dataset objects which can be passes to torch's DataLoader as the dataset.
'''    
def InitializeDatasets(datasetParams, enableShifting):
    trainDataset = SNP_Dataset(featherFilePath = datasetParams.get("complete_dataset_file_path"),
                                   phenoFilePath = datasetParams.get("pheno_path"),
                                   idFilePath = datasetParams.get("train_IDs"), 
                                   snpSubsetPath = datasetParams.get("SNP_set"),
                                   validMafSnpsPath = datasetParams.get("consistent_maf_SNPs"),
                                   vaderPatchMappingPath = datasetParams.get("patch_mapping_path"),
                                   enableShifting = enableShifting,
                                   trainingSetAgeStatsPath = datasetParams.get("age_train_stats"), 
                                   cacheWritePath = datasetParams.get("cached_training_feather_path"), #Need a unique path for train and val otherwise validation will register as preprocessed given existence of featurePatchSizes
                                   sparsePatchThreshold = datasetParams.get("sparse_patch_threshold"))
        
    validationDataset = SNP_Dataset(featherFilePath = datasetParams.get("complete_dataset_file_path"),
                                   phenoFilePath = datasetParams.get("pheno_path"),
                                   idFilePath = datasetParams.get("val_IDs"), 
                                   snpSubsetPath = datasetParams.get("SNP_set"),
                                   validMafSnpsPath = datasetParams.get("consistent_maf_SNPs"),
                                   vaderPatchMappingPath = datasetParams.get("patch_mapping_path"),
                                   enableShifting = enableShifting,
                                   trainingSetAgeStatsPath = datasetParams.get("age_train_stats"), 
                                   cacheWritePath = datasetParams.get("cached_val_feather_path"),
                                   sparsePatchThreshold = datasetParams.get("sparse_patch_threshold"))

    return trainDataset, validationDataset

'''
About: Method to initialize a config specified optimizer.

Inputs: 1) optimizer_name: String corresponding to desired optimizer
        2) model_params: Model parameters to be passed to the optimizer
        3) learning_rate: Float corresponding to the desired learning rate.
Output: A torch optimizer corresponding to specified optimizer_name 
'''
def GetOptimizer(optimizer_name, model_params, learning_rate, **kwargs):
    optimizerOptions = {"AdamW": optim.AdamW,
                        "Adam": optim.Adam, 
                        "SGD": optim.SGD,
                        "RMSprop": optim.RMSprop}
    
    if optimizer_name in optimizerOptions:
        return optimizerOptions[optimizer_name](model_params, lr = learning_rate, **kwargs)
    
    else:
        raise ValueError(f"Optimizer '{optimizer_name}' invalid. Valid optimizer_name options are: 'AdamW', 'Adam', 'SGD', and 'RMSprop'.")

'''
About: Method to initialize a VADEr model and optimizer according to specified input parameters.

Inputs: 1) modelParams: Dictionary of model parameters loaded from config.
        2) patch_sizes: Dictionary of patch sizes - needed to initialize VADEr patch projection dimensions. 
        3) learning_rate: Float corresponding to config specified learning rate.
        4) optimizer_name: String corresponding to an optimizer
        5) training_objective: String corresponding to specified trainingObjective

Outputs: 1) lordVader: VADEr model
         2) optimizer: Optimizer with relevant model parameters. Configured according to given optimizer_name learning rate and additional 
                       specified keyword arguments.
'''
def InitializeModelAndOptimizer(modelParams, patch_sizes, learning_rate, optimizer_name, training_objective, **kwargs):
    if training_objective == "sup_con":
        contrastive_projection_net_dim = modelParams["contrastive_projection_net_dim"]

    else:
        contrastive_projection_net_dim = None

    attention = modelParams.get("attention")
    patchLayerNorm = modelParams.get("patch_layer_norm")
    numRegisters = modelParams.get("num_registers")

    if attention is None:
        attention = "MHA"

    if patchLayerNorm is None:
        patchLayerNorm = attention == "LSA"

    if numRegisters is None:
        numRegisters = 0

    lordVader = VADEr(patchSizes = patch_sizes,
                      modelDim = modelParams["model_dim"],
                      mlpDim = modelParams["model_dim"] * modelParams["mlp_scale"],
                      depth = modelParams["num_transformer_blocks"],
                      attnHeads = modelParams["num_attention_heads"],
                      attnHeadDim = modelParams["model_dim"]//modelParams["num_attention_heads"],
                      multitaskOutputs = modelParams["prediction_dims"],
                      clumpProjectionDropout = modelParams["patch_projection_dropout"],
                      dropout = modelParams["model_dropout"], 
                      ageInclusion = modelParams["age_inclusion"],
                      aggr = modelParams["aggregation"],
                      context = modelParams.get("cls_representation"),
                      patchProjectionActivation = modelParams["non_linear_patch_projection"],
                      patchLayerNorm = patchLayerNorm, 
                      trainingObjective = training_objective,
                      attention = attention,
                      numRegisters = numRegisters,
                      ffActivation = modelParams.get("mlp_method"), 
                      contrastive_projection_net_dim = contrastive_projection_net_dim)
    
    if modelParams.get("load_pretrained"): #Fit a linear classifier head on top of a sup_con pre-trained VADEr model.
        logging.info("Loading pretrained VADEr (contrastive-loss) model and freezing weights...")
        
        # Load pretrained model
        lordVader.load_state_dict(torch.load(modelParams.get("load_pretrained"))["modelStateDict"])
        
        # Freeze model weights:
        for param in lordVader.parameters():
            param.requires_grad = False
        
        # Discard projection network (as self-supervised contrastive learning is now complete) and add trainable linear classifier
        lordVader.projectionNetwork = nn.Linear(modelParams["model_dim"], modelParams["prediction_dims"]["disease"])
    
    optimizer = GetOptimizer(optimizer_name = optimizer_name, model_params = lordVader.parameters(), learning_rate = learning_rate, **kwargs)
    
    return lordVader, optimizer

def main(args):
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    DDP_Setup()
    torch.manual_seed(3)

    #Pull specs from config:
    config = LoadConfig(args.config_path)
    trainAndCheckpointSpecs = config["train_and_checkpoint"]
    modelParams = config["model_params"]
    optimizerParams = config["optimizer_params"]
    datasetParams = config["dataset"]

    if trainAndCheckpointSpecs.get("scheduler"):
        assert (
            trainAndCheckpointSpecs.get("scheduler") in {"cosine_with_warmup", "linear_with_warmup", "cosine_with_hard_restarts_and_warmup", "None"} 
        ), f'Invalid specified scheduler: {trainAndCheckpointSpecs.get("scheduler")}. Scheduler if specified must be in {"cosine_with_warmup", "linear_with_warmup", "cosine_with_hard_restarts_and_warmup", "None"}.'

    #initialize dataset on main process and add barrier
    gpuID = int(os.environ["LOCAL_RANK"])

    logger.info(f"SPT enabled: {trainAndCheckpointSpecs['shift_patches']}")

    if gpuID != 0: #Ensure only main/first process in distributed training pre-processes/caches the dataset for memory efficiency 
        logger.info(f"[GPU{gpuID}] waiting for pre-processing/caching of dataset by main process...")
        torch.distributed.barrier()

        trainDataset, validationDataset = InitializeDatasets(datasetParams = datasetParams, enableShifting = trainAndCheckpointSpecs["shift_patches"])
    

    else: #have main/rank 0 process handle caching for all other processes to use
        logger.info(f"[GPU{gpuID}] Beginning caching/preprocessing...")

        trainDataset, validationDataset = InitializeDatasets(datasetParams = datasetParams, enableShifting = trainAndCheckpointSpecs["shift_patches"])
        
        logger.info(f"[GPU{gpuID}] Completed caching... waiting for cached loading by other processes")

        torch.distributed.barrier()
        
    patchSizes = trainDataset.patchSizes

    #Generate chromosome mask if positional mapping provided in datasetParams:
    chromosomeMask = None

    if (trainAndCheckpointSpecs.get("masking") == "chrom") and (datasetParams.get("patch_to_chrom_mapping") is not None):
        logger.info("Enabling chromosome masking")
        chromosomeMask = GenerateChromosomePatchMask(patch_to_chrom_mapping_path = datasetParams.get("patch_to_chrom_mapping"),
                                                     feature_patches = patchSizes,
                                                     num_registers = modelParams.get("num_registers"))


    trainLoader = PrepareDistributedLoader(dataset = trainDataset, shuffle = True, batch_size = trainAndCheckpointSpecs.get("batch_size"), num_workers = trainAndCheckpointSpecs.get("num_workers"))
    valLoader = PrepareDistributedLoader(dataset = validationDataset, shuffle = False, batch_size = trainAndCheckpointSpecs.get("batch_size"), num_workers = trainAndCheckpointSpecs.get("num_workers")) 

    #when steps selected needed to pass in a trainEvalLoader and establish the number of training epochs
    if trainAndCheckpointSpecs.get("checkpoint_freq_method") == "Steps": 
        logger.info("Employing STEP-based training and checkpointing")
        checkpointFreqMethod = trainAndCheckpointSpecs.get("checkpoint_freq_method")
        trainEvalLoader = PrepareDistributedLoader(dataset = trainDataset, shuffle = False, batch_size = trainAndCheckpointSpecs.get("batch_size"), num_workers = trainAndCheckpointSpecs.get("num_workers"))
        trainingEpochs = StepsToEpochs(maxNumberSteps = trainAndCheckpointSpecs["num_training_steps"], 
                                       stepsPerEpoch = len(trainLoader),
                                       gradientAccumulationSteps = trainAndCheckpointSpecs.get("gradient_accumulation"))
        
        if isinstance(trainAndCheckpointSpecs.get("scheduler_warmup_steps"), (int, float)):
            schedulerWarmupSteps = int(trainAndCheckpointSpecs.get("scheduler_warmup_steps"))
            totalTrainingSteps = int(trainAndCheckpointSpecs["num_training_steps"])

            assert schedulerWarmupSteps < totalTrainingSteps, f"""Provided scheduler_warmup_steps {trainAndCheckpointSpecs.get("scheduler_warmup_steps")} >= the total number of 
                                                                                                 training steps specified: {trainAndCheckpointSpecs["num_training_steps"]} 
                                                                                                 (after conversion to integers if necessary)."""
    else:
        checkpointFreqMethod = "Epochs"
        trainEvalLoader = None
        trainingEpochs = trainAndCheckpointSpecs["num_training_steps"]
        totalTrainingSteps = EpochsToSteps(epochs = trainingEpochs, 
                                           stepsPerEpoch = len(trainLoader),
                                           gradientAccumulationSteps = trainAndCheckpointSpecs.get("gradient_accumulation"))

        if isinstance(trainAndCheckpointSpecs.get("scheduler_warmup_steps"), (int, float)):
            assert trainAndCheckpointSpecs.get("scheduler_warmup_steps") < totalTrainingSteps, f"""Provided scheduler_warmup_steps {trainAndCheckpointSpecs.get("scheduler_warmup_steps")} >= the total number of 
                                                                                                 training steps for the specified number of epochs: {trainingEpochs}. For {trainingEpochs} epochs there are {totalTrainingSteps} 
                                                                                                 total training steps."""
            schedulerWarmupSteps = int(trainAndCheckpointSpecs.get("scheduler_warmup_steps"))

    trainingObjective = "cross_entropy" if "cross_entropy" in trainAndCheckpointSpecs["loss"] else trainAndCheckpointSpecs["loss"]

    model, optimizer = InitializeModelAndOptimizer(modelParams = modelParams, 
                                                   patch_sizes = patchSizes, 
                                                   learning_rate = float(trainAndCheckpointSpecs["lr"]),
                                                   optimizer_name = optimizerParams["optimizer"],
                                                   training_objective = trainingObjective,
                                                   **{k:v for k,v in optimizerParams.items() if k != "optimizer"})
    
    #Handle scheduler (potentially move to a separate function - initialize scheduler?)
    if trainAndCheckpointSpecs.get("scheduler") == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(optimizer = optimizer,
                                                    num_warmup_steps = schedulerWarmupSteps,
                                                    num_training_steps = totalTrainingSteps)
    elif trainAndCheckpointSpecs.get("scheduler") == "linear_with_warmup":
        scheduler = get_linear_schedule_with_warmup(optimizer = optimizer,
                                                    num_warmup_steps = schedulerWarmupSteps,
                                                    num_training_steps = totalTrainingSteps)
    elif trainAndCheckpointSpecs.get("scheduler") == "cosine_with_hard_restarts_and_warmup":
        numRestarts = trainAndCheckpointSpecs.get("num_restarts")
        if not isinstance(numRestarts, (int, float)):
            logger.warning("Invalid 'num_restarts'/'num_restarts' not provided with 'cosine_with_hard_restarts_and_warmup' schedule. Defaulting to 1.")
            numRestarts = 1
        elif numRestarts < 1:
            logger.warning(f"Provided 'num_restarts' of {numRestarts} is invalid with 'cosine_with_hard_restarts_and_warmup' schedule. 'num_restarts' >= 1. 'Setting num_restarts' to 1.")
            numRestarts = 1
        else:
            numRestarts = int(numRestarts)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer = optimizer,
                                                                       num_warmup_steps = schedulerWarmupSteps,
                                                                       num_training_steps = totalTrainingSteps,
                                                                       num_cycles = numRestarts)

    else:
        scheduler = get_constant_schedule(optimizer = optimizer)

    #Handle custom regularization inclusion (e.g., patchDropout)
    enable_patch_dropout = trainAndCheckpointSpecs.get("enable_patch_dropout")
    if enable_patch_dropout is None:
        enable_patch_dropout = False 

    #Initialize Trainer Ash Ketchum (from Pallet Town...)
    askKetchum = Trainer(model = model,
                         trainLoader = trainLoader,
                         valLoader = valLoader, 
                         optimizer = optimizer,
                         scheduler = scheduler, 
                         checkpointPath = trainAndCheckpointSpecs["model_checkpoint_path"],
                         checkpointFreq = trainAndCheckpointSpecs["checkpoint_frequency"],
                         trainingSummaryPath = trainAndCheckpointSpecs["training_summary_path"],
                         trainingObjective = trainAndCheckpointSpecs["loss"],
                         checkpointFreqMethod = checkpointFreqMethod,
                         gradientClipping = trainAndCheckpointSpecs.get("gradient_clipping"),
                         gradientAccumulation = trainAndCheckpointSpecs.get("gradient_accumulation"),
                         patchDropout = enable_patch_dropout,
                         mask = chromosomeMask,
                         trainEvalLoader = trainEvalLoader)


    #Become a pokemon SITH lord master:
    askKetchum.Train(numberTrainingEpochs = trainingEpochs, 
                     checkpointK = trainAndCheckpointSpecs["k_checkpoint"], 
                     checkpointMetric = trainAndCheckpointSpecs["checkpoint_metric"],
                     trainingSteps = totalTrainingSteps,
                     mixup_method = trainAndCheckpointSpecs.get("mixup_method"),
                     mixup_alpha = trainAndCheckpointSpecs.get("mixup_alpha"),
                     label_smoothing = trainAndCheckpointSpecs.get("label_smoothing"))
    
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group("Input/output options:")
    path_args.add_argument('--config_path', type=str, help='Path to config .yaml file to train a VADEr model')
    args = parser.parse_args()

    main(args)
