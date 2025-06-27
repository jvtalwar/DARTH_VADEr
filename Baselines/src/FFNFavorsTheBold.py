# -*- coding: utf-8 -*-
"""
@author: James V. Talwar

About: FFNFavorsTheBold.py takes a config and performs either MULTI-TASK (Disease and Ethnicity) or SINGLE-TASK (Disease only) training for either a Fully-Connected Feed 
Forward Network (FCFFN) config specified single model configuration or performs TPE hyperoptimization (for a specified number of attempts). CSV files documenting model 
performance and model checkpointing are performed and saved in the paths specified in the config file. For hyperoptimization, studies are saved in rdb format which 
allows for multi-gpu training and for resumption of studies w/o having to start from scratch. 

Example Usage: python ./FFNFavorsTheBold.py --config_path ../Configs/DRIVE/5e-8/LD_MTL_5e-8.yaml --training_objective MTL/ST

"""

import yaml 
import argparse 
import torch 
import os
import sys
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from InSNPtion import FCFFN 
import torch.nn as nn
import torch.optim as optim
from optimizer import Lookahead
from radam import RAdam 
import torch.nn.functional as F
from AUC_Utils import Generate_FFN_Preds, Calc_ROC
from NeverArgueWithTheDataloader import *  
from torch.autograd import Variable
import optuna
import joblib

import logging
logger = logging.getLogger(__name__)


torch.autograd.set_detect_anomaly(True) #Explicitly raises an error if invalid operations occur

def load_config(path):
    """
    Load the configuration from Cosmo.yaml.
    """
    return yaml.load(open(path, 'r'), Loader=yaml.SafeLoader)

'''
About: Calculate accuracy numerator for multi-class prediction (i.e., ethnicity accuracy)
Input: of 2 tensors --> make sure labels is a tensor of size == batch size
Output: Total number of matches (accuracy can be returned by removing the comment # below); 
'''
def AccuracyCalc(raw, labels): 
    payMeForMyPredictions = torch.argmax(F.softmax(raw, dim = 1), dim = 1)
    return sum(payMeForMyPredictions.eq(labels)).item() #/len(lables)

'''
About: Calculate disease accuracy numerator (i.e., number of correct predictions for binary class prediction)
Input(s): 1) raw_preds: Unactivated output of the model (i.e., logits) 
          2) labels: Tensor of labels (i.e., 0 or 1)
Output:  Total number of matches (accuracy can be computed by normalizing across the total number of individuals)
'''
def Correct_Preds(raw_preds,labels):
    predictions = torch.round(torch.sigmoid(raw_preds))
    return sum(predictions.eq(labels)).item()


'''
VALIDATION:
Input(s): 1) model 2) current epoch number 3) validation dataloader 4) Loss function 5) device
6) Weighted Task for losses (type dict)
Output(s): 4 floats
1) Validation Accuracy 2) Ethnicity Validation Accuracy 3) Family History Validation Accuracy 4) Total Validation (weighted) Loss
'''
def Validate(model, epochNum, data, lossFn, device, weightedLoss, learningObjective, ageInclusion):
    model.eval()
    totalSamples = 0
    totalLoss = 0
    numerator = 0
    ethnAcc = 0
    
    with torch.no_grad():
        for i, (snpBatch, pcBatch, ethnBatch, fHBatch, zAgeBatch) in enumerate(data):
            
            totalSamples += len(pcBatch) #running total of number of samples

            #move labels to gpu 
            snpBatch = snpBatch.to(device)
            pcBatch = pcBatch.to(device)
            ethnBatch = ethnBatch.to(device)
            #fHBatch = fHBatch.to(device)

            #Run it: recall output returns a list to handle multi-task learning so here the prediction is output[0],[1], ...
            if ageInclusion: #including age - need to pass in more than genotypes
                output = model(x = snpBatch, ageBatch = zAgeBatch.to(device))
            else:
                output = model(x = snpBatch)

            if learningObjective == "MTL":
                loss = weightedLoss['PC']*lossFn[0](output[0], pcBatch.float()) + weightedLoss['Ethn']*lossFn[1](output[1], ethnBatch.squeeze(1).long()) 
            else:
                loss = lossFn[0](output[0], pcBatch.float())

            #Calculate accuracy:
            numerator += Correct_Preds(output[0], pcBatch)
            if len(output) > 1:
                ethnAcc += AccuracyCalc(output[1], ethnBatch.squeeze(1))

            totalLoss += loss.item()
             
        
    #assert totalSamples == len(data.dataset), "Validation set size does not match the total number of samples processed. Check dataloader and dataset." 
        
    epochAcc = numerator/(totalSamples)
    ethnicityAccuracy = ethnAcc/totalSamples
    
    totalLoss = totalLoss/totalSamples
    
    return epochAcc, ethnicityAccuracy, totalLoss 

'''
TRAIN:

Input(s): 1) Model 2) Current Epoch Number 3) training dataloader 4) loss functions (type list) 
5) the device available (cpu or gpu/cuda) 6) Weighted Task for losses (type dict) 7) Optimizer
8) learningObjectiove == {"MTL", "ST"} 9) ageInclusion - boolean about whether to pass age into model  

Output(s): 4 floats
  1) Training Accuracy 2) Ethnicity Training Accuracy 3) Family History Training Accuracy 4) Total training (weighted) Loss
'''

def Train(model, epochNum, data, lossFn, device, weightedLoss, optimizer, learningObjective, ageInclusion):
    model.train() #Put model in train mode (i.e., Dropout, and batch norm are on)
    numerator = 0 
    ethnAcc = 0
    totalSamples = 0
    totalLoss = 0
        
    for i, (snpBatch, pcBatch, ethnBatch, fHBatch, zAgeBatch) in enumerate(data):
        totalSamples += len(pcBatch) #running total of number of samples
        
        optimizer.zero_grad() #zero the gradient to prevent accumulation; can also do model.zero_grad() 
        
        #move labels to gpu 
        snpBatch = snpBatch.to(device)
        pcBatch = pcBatch.to(device)
        ethnBatch = ethnBatch.to(device)
        #fHBatch = fHBatch.to(device)
        
        #Run it:
        if ageInclusion: #including age - need to pass in more than genotypes
            output = model(x = snpBatch, ageBatch = zAgeBatch.to(device))
        else:
            output = model(x = snpBatch)
        
        if learningObjective == "MTL":
            loss = weightedLoss['PC']*lossFn[0](output[0], pcBatch.float()) + weightedLoss['Ethn']*lossFn[1](output[1], ethnBatch.squeeze(1).long()) 
        else:
            loss = lossFn[0](output[0], pcBatch.float())
        
        #Calculate accuracy:
        numerator += Correct_Preds(output[0], pcBatch)
        if len(output) > 1:
            ethnAcc += AccuracyCalc(output[1], ethnBatch.squeeze(1))
            
        totalLoss += loss.item()
        
        #Manage Memory:
        del snpBatch, pcBatch, ethnBatch
        torch.cuda.empty_cache()
        
        #Train it: 
        loss.backward()
        optimizer.step()

    #assert totalSamples == len(data.dataset), "Train set size does not match the total number of samples processed. Check dataloader and dataset." 
        
    epochAcc = numerator/(totalSamples)
    ethnEpochAcc = ethnAcc/totalSamples
    
    totalLoss = totalLoss/totalSamples
    
    return epochAcc, ethnEpochAcc, totalLoss  


def Train_Hyperoptimize(trial, config, trainingObjective): 
    generalSpecs = config["General"]
    modelParams = config["ModelParams"]
    dataLoaderParams = config["Dataloader"]
    
    scaleFactor = modelParams.get("H1_Dimension_Upper_Bound")
    
    if generalSpecs["hyperOptimize"]:
        logger.info("Running Trial {} ... \n".format(trial.number))
    
    #Set device (ideally GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Instantiate dataloaders    
    trainLoader = get_loader(feather = dataLoaderParams.get("FeatherPath"),
                             phenos = dataLoaderParams.get("PhenoPath"),
                             ids = dataLoaderParams.get("TrainIDs"),
                             snps = dataLoaderParams.get("SNP_Set"),
                             id_column = "#IID",
                             ageTrainStats = dataLoaderParams.get("AgeTrainStats"),
                             batch_size = generalSpecs.get("batchSize"),
                             validMafSnpPath = dataLoaderParams.get("ConsistentMafSNPs"),
                             featherWritePath = dataLoaderParams.get("FeatherWritePath"),
                             shuffle = True,
                             num_workers = generalSpecs.get("numWorkers")) 
    
    valLoader = get_loader(feather = dataLoaderParams.get("FeatherPath"), 
                           phenos = dataLoaderParams.get("PhenoPath"), 
                           ids = dataLoaderParams.get("ValIDs"), 
                           snps = dataLoaderParams.get("SNP_Set"), 
                           id_column = "#IID",
                           ageTrainStats = dataLoaderParams.get("AgeTrainStats"),
                           batch_size = generalSpecs.get("batchSize"), 
                           validMafSnpPath = dataLoaderParams.get("ConsistentMafSNPs"),
                           featherWritePath = dataLoaderParams.get("FeatherWritePath"),
                           shuffle = True, 
                           num_workers = generalSpecs.get("numWorkers"))  
    
    #when scaleFactor = None (i.e., no explicit upper bound on hidden layer dimension) default upper bound of model to the input dimension size
    if scaleFactor is None: 
        scaleFactor = len(trainLoader.dataset[0][0])
    
    print("Using SNP set {} with feature size: {}".format(dataLoaderParams.get("SNP_Set"), len(trainLoader.dataset[0][0])))
    
    #Training metrics tracking 
    losses = []
    accuracies = []
    ethnicityAccuracies = []
    trainROCs = []
    
    #Validation metrics tracking 
    valAccs = []
    valEthnicityAccuracies = []
    valLosses = []
    valROCs = []
    
    #Hyperoptimization vs Single Run Block:
    if generalSpecs["hyperOptimize"]:
        #Learning rate
        lr = trial.suggest_float("lr", 1e-6, 1e-1, log = True)

        #model hyper parameters
        numberOfHiddenLayers = trial.suggest_int("numHiddenLayers", 1, generalSpecs['maxLayers'])
        thick = [] #layer widths
        minWidth = 256 
        
        #populate widths in a monotonically decreasing manner
        for i in range(numberOfHiddenLayers):
            if len(thick) == 0:
                thick.append(trial.suggest_int("n_units_l{}".format(i), minWidth, scaleFactor, step = minWidth)) 
            else:
                thick.append(trial.suggest_int("n_units_l{}".format(i), min(minWidth, thick[len(thick)-1]), max(minWidth, thick[len(thick)-1]), step = minWidth))
        
        withTheDropout = trial.suggest_float("dropout", 0.0, 0.8, step = 0.1)
        activateOrder66 = trial.suggest_categorical("Activation_Function", ["ReLU", "Mish", "GELU"])
        
        ffn_model = FCFFN(inputDimension = len(trainLoader.dataset[0][0]), 
                             numLayers = numberOfHiddenLayers, 
                             layerWidths = thick, 
                             multitaskOutputs = modelParams["multiTaskOutputs"], 
                             dropout = withTheDropout, 
                             activation = activateOrder66, 
                             ageInclusion = modelParams["ageInclusion"])
        
        optimizer = optim.AdamW(ffn_model.parameters(), lr = lr)
        optimizer_name = optimizer.__class__.__name__
        print("Using optimizer {} \n".format(optimizer_name))
        
        ffn_model = ffn_model.to(device)

        if torch.cuda.device_count() > 1: #if more than one GPU is visible to Optuna run in parallel (with DataParallel)
            logger.warning("\n Attempting to distribute FC-FFN model across {} GPUs...\n".format(torch.cuda.device_count()))
            ffn_model = torch.nn.DataParallel(ffn_model, list(range(torch.cuda.device_count())))
        
        if str(device) != 'cuda':
            logger.warning("GPU not detected - training on CPU.")
        
        if modelParams["weightedLoss"]:
            disease_weighting = torch.FloatTensor(modelParams['weights']['PC']).to(device) 
            ancestry_weighting = torch.FloatTensor(modelParams['weights']["Ethnicity"]).to(device)
            criterion = [nn.BCEWithLogitsLoss(weight=disease_weighting), nn.CrossEntropyLoss(weight=ancestry_weighting)] 
            print("Using Weighted Cross Entropy as Loss f(x)")
                    
        else:
            criterion = [nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()] #,nn.BCEWithLogitsLoss()] 
            print("Using Unweighted Cross Entropy as Loss f(x)")
            
        logger.info("Finished model initialization... beginning hYpErOpTiMiZeD training and validation \n\n")
        
        #optimize task specific weighting 
        if trainingObjective == "MTL":
            weightedTaskPC = trial.suggest_int("PCWeight", 1, 100)/100
            weightedTaskEthnicity = 1 - weightedTaskPC #Balance 
            completeWeightedTask = {"PC":weightedTaskPC, "Ethn":weightedTaskEthnicity} 
            logger.info(f"Optuna trial {trial.number} loss weighting: {completeWeightedTask}")
        
        else:
            completeWeightedTask = {"PC": 0.5, "Ethn": 0.5} #isn't used for ST - but passed in as a param
        
    else: 
        ffn_model = FCFFN(inputDimension = len(trainLoader.dataset[0][0]), 
                             numLayers = len(modelParams["layerWidths"]), 
                             layerWidths = modelParams["layerWidths"], 
                             multitaskOutputs = modelParams["multiTaskOutputs"], 
                             dropout = modelParams["dropout"], 
                             activation = modelParams["activation"], 
                             ageInclusion = modelParams["ageInclusion"])
        
        
        optimizerMapping = {"AdamW": optim.AdamW(ffn_model.parameters(), lr = float(generalSpecs["learningRate"])),
                            "Adam": optim.Adam(ffn_model.parameters(), lr = float(generalSpecs["learningRate"])), 
                            "LoneRanger": Lookahead(optimizer = optim.Adam(ffn_model.parameters(), lr = float(generalSpecs["learningRate"])), k=5, alpha =0.5),
                            "Ranger": Lookahead(optimizer = RAdam(ffn_model.parameters(), lr = float(generalSpecs["learningRate"])), k=5, alpha =0.5),
                            "SGD": optim.SGD(ffn_model.parameters(), lr = float(generalSpecs["learningRate"]))}
        
        optimizer = optimizerMapping.get(generalSpecs["optimizer"])
        print("Using optimizer {} \n".format(optimizer.__class__.__name__))
        
        ffn_model = ffn_model.to(device)

        if torch.cuda.device_count() > 1: #if more than one GPU is visible to Optuna run in parallel (with DataParallel)
            logger.warning("\n Attempting to distribute model across {} GPUs...\n".format(torch.cuda.device_count()))
            ffn_model = torch.nn.DataParallel(ffn_model, list(range(torch.cuda.device_count())))
                
        if str(device) != 'cuda':
            logger.warning("GPU not detected - training on CPU.")
        
        #whether want to implement class weight loss
        if modelParams["weightedLoss"]:
            disease_weighting = torch.FloatTensor(modelParams['weights']['PC']).to(device) 
            ancestry_weighting = torch.FloatTensor(modelParams['weights']["Ethnicity"]).to(device)
            criterion = [nn.BCEWithLogitsLoss(weight=disease_weighting), nn.CrossEntropyLoss(weight=ancestry_weighting)] 
            print("Using Weighted Cross Entropy as Loss f(x)")
                    
        else:
            criterion = [nn.BCEWithLogitsLoss(),nn.CrossEntropyLoss()] #,nn.BCEWithLogitsLoss()] 
            print("Using Unweighted Cross Entropy as Loss f(x)")
        
        logger.info("Finished model initialization... beginning training and validation \n\n")
    
    #Single Model Baseline Metrics
    if generalSpecs["hyperOptimize"] == False:
        #TRAIN 0:
        trainZeroAcc,trainZeroEthn,trainZeroLoss = Validate(model=ffn_model, epochNum=0, data=trainLoader, lossFn=criterion, device=device, weightedLoss = modelParams["weightedTask"], learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
        accuracies.append(trainZeroAcc)
        losses.append(trainZeroLoss)
        ethnicityAccuracies.append(trainZeroEthn)
        
        #AUCs
        whatAreMyScores,phenotypeLabels = Generate_FFN_Preds(model = ffn_model, loader = trainLoader, device = device, ageInclusion = modelParams["ageInclusion"])
        trainROCs.append(Calc_ROC(whatAreMyScores, phenotypeLabels))
        
        del trainZeroAcc, trainZeroLoss,trainZeroEthn,whatAreMyScores,phenotypeLabels 

        #VALIDATION 0:
        valZeroAcc, valEthnZeroAcc,valZeroLoss = Validate(model = ffn_model, epochNum=0, data=valLoader, lossFn=criterion, device=device, weightedLoss = modelParams["weightedTask"], learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
        valAccs.append(valZeroAcc)
        valEthnicityAccuracies.append(valEthnZeroAcc)
        valLosses.append(valZeroLoss)
        
        #AUCs
        whatAreMyScores,phenotypeLabels = Generate_FFN_Preds(model = ffn_model, loader = valLoader, device = device, ageInclusion = modelParams["ageInclusion"])
        valROCs.append(Calc_ROC(whatAreMyScores, phenotypeLabels))
        
        del valZeroAcc,valEthnZeroAcc,valZeroLoss,whatAreMyScores,phenotypeLabels 
    
    #Hyperoptimization baseline metrics
    else:
        #TRAIN 0: 
        trainZeroAcc,trainZeroEthn,trainZeroLoss = Validate(model = ffn_model, epochNum = 0, data = trainLoader, lossFn = criterion, device = device, weightedLoss = completeWeightedTask, learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
        accuracies.append(trainZeroAcc)
        losses.append(trainZeroLoss)
        ethnicityAccuracies.append(trainZeroEthn)
        
        whatAreMyScores,phenotypeLabels = Generate_FFN_Preds(model = ffn_model, loader = trainLoader, device = device, ageInclusion = modelParams["ageInclusion"])
        trainROCs.append(Calc_ROC(whatAreMyScores, phenotypeLabels))
       
        del trainZeroAcc,trainZeroLoss,trainZeroEthn,whatAreMyScores,phenotypeLabels
        
        #VALIDATION 0:
        valZeroAcc, valEthnZeroAcc,valZeroLoss = Validate(model = ffn_model, epochNum = 0, data = valLoader, lossFn = criterion, device = device, weightedLoss = completeWeightedTask, learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
        valAccs.append(valZeroAcc)
        valEthnicityAccuracies.append(valEthnZeroAcc)
        valLosses.append(valZeroLoss)
        
        #AUCs
        whatAreMyScores,phenotypeLabels = Generate_FFN_Preds(model = ffn_model, loader = valLoader, device = device, ageInclusion = modelParams["ageInclusion"])
        valROCs.append(Calc_ROC(whatAreMyScores, phenotypeLabels))
        del valZeroAcc,valEthnZeroAcc,valZeroLoss,whatAreMyScores,phenotypeLabels
        

    for epoch in range(generalSpecs["epochs"]):  
        #Train:
        if (generalSpecs["hyperOptimize"]):
            epochAcc, ethnEpochAcc,totalLoss = Train(model = ffn_model, epochNum=epoch+1, data=trainLoader, lossFn=criterion, device=device, weightedLoss=completeWeightedTask, optimizer=optimizer, learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])

        else:
            epochAcc, ethnEpochAcc,totalLoss = Train(model = ffn_model, epochNum=epoch+1, data=trainLoader, lossFn=criterion, device=device, weightedLoss=modelParams["weightedTask"], optimizer=optimizer, learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
        
        accuracies.append(epochAcc)
        ethnicityAccuracies.append(ethnEpochAcc)
        losses.append(totalLoss)
        
        #Calc_AUC on the training set now after training:
        whatAreMyScores,phenotypeLabels = Generate_FFN_Preds(model = ffn_model, loader = trainLoader, device = device, ageInclusion = modelParams["ageInclusion"])
        trainROCs.append(Calc_ROC(whatAreMyScores, phenotypeLabels))
      
        #Validate:
        if (generalSpecs["hyperOptimize"]):
            if (optimizer_name == "LoneRanger") or (optimizer_name == "Ranger"):
                optimizer._backup_and_load_cache()
                valAcc,valEthnAcc,valLoss = Validate(model = ffn_model, epochNum = epoch+1, data = valLoader, lossFn = criterion, device = device, weightedLoss = completeWeightedTask, learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
                optimizer._clear_and_load_backup()
            
            else:
                valAcc,valEthnAcc,valLoss = Validate(model = ffn_model, epochNum = epoch+1, data = valLoader, lossFn = criterion, device = device, weightedLoss = completeWeightedTask, learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
            
        else:
            if (generalSpecs["optimizer"] == "LoneRanger") or (generalSpecs["optimizer"] == "Ranger"):
                optimizer._backup_and_load_cache()
                valAcc,valEthnAcc,valLoss = Validate(model = ffn_model, epochNum = epoch+1, data = valLoader, lossFn = criterion, device=device, weightedLoss=modelParams["weightedTask"], learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
                optimizer._clear_and_load_backup()
            else:
                valAcc,valEthnAcc,valLoss = Validate(model = ffn_model, epochNum = epoch+1, data = valLoader, lossFn = criterion, device = device, weightedLoss = modelParams["weightedTask"], learningObjective = trainingObjective, ageInclusion = modelParams["ageInclusion"])
        
        valAccs.append(valAcc)
        valEthnicityAccuracies.append(valEthnAcc)
        valLosses.append(valLoss)
        
        
        #Calc AUC on the validation set:
        whatAreMyScores, phenotypeLabels = Generate_FFN_Preds(model = ffn_model, loader = valLoader, device = device, ageInclusion = modelParams["ageInclusion"])
        currentValROC = Calc_ROC(whatAreMyScores, phenotypeLabels)
        valROCs.append(currentValROC)
        
        
        #Save trained models: (if want every five unmute line below and add indent to corresponding if else statements)
        #if ((epoch+1) % 5 == 0): #save models every 5 epochs; +1 needed to make everything 1 indexed
        if generalSpecs["hyperOptimize"]:
            try: #handle saving when wrapped in DataParallel
                torch.save(ffn_model.module.state_dict(), '{}.pt'.format(os.path.join(generalSpecs["modelCheckpointingPath"], "Optuna_Study_Trial_" + str(trial.number) + "_Epoch_" + str(epoch+1))))
            except AttributeError:
                torch.save(ffn_model.state_dict(), '{}.pt'.format(os.path.join(generalSpecs["modelCheckpointingPath"], "Optuna_Study_Trial_" + str(trial.number) + "_Epoch_" + str(epoch+1))))
            
        else:
            try:
                torch.save(ffn_model.module.state_dict(), '{}.pt'.format(os.path.join(generalSpecs["modelCheckpointingPath"], generalSpecs["studyName"] + "_Epoch_" + str(epoch+1))))
            except AttributeError:
                torch.save(ffn_model.state_dict(), '{}.pt'.format(os.path.join(generalSpecs["modelCheckpointingPath"], generalSpecs["studyName"] + "_Epoch_" + str(epoch+1))))

        if generalSpecs['hyperOptimize']:
            trial.report(currentValROC, epoch+1) #hyperoptimize for validation set AUC
            if trial.should_prune():
                goingOut = pd.DataFrame([accuracies, ethnicityAccuracies, trainROCs, losses, valAccs, valEthnicityAccuracies, valROCs,valLosses]).T
                goingOut.columns = ['Train_PC_Acc', "Train_Ethn_Acc", 'Train_AUC','Train_Loss','Val_PC_Acc', 'Val_Ethn_Acc','Val_AUC','Val_Loss']
                
                goingOut.to_csv(os.path.join(generalSpecs['pathForCSVSummary'], "Optuna_Study_Trial_" + str(trial.number) + ".tsv"), sep = "\t")
                
                print("Pruning Trial {} at epoch {} \n:".format(trial.number, epoch + 1))
                
                raise optuna.exceptions.TrialPruned()
          
    
    if generalSpecs["hyperOptimize"] == False:
        return accuracies, ethnicityAccuracies, trainROCs, losses, valAccs, valEthnicityAccuracies, valROCs, valLosses
    
    else:
        goingOut = pd.DataFrame([accuracies, ethnicityAccuracies, trainROCs, losses, valAccs, valEthnicityAccuracies, valROCs, valLosses]).T
        
        goingOut.columns = ['Train_PC_Acc', "Train_Ethn_Acc", 'Train_AUC','Train_Loss','Val_PC_Acc', 'Val_Ethn_Acc', 'Val_AUC','Val_Loss'] 
        
        goingOut.to_csv(os.path.join(generalSpecs['pathForCSVSummary'], "Optuna_Study_Trial_" + str(trial.number) + ".tsv"), sep = '\t')        
        print("Completed Trial {}: \n".format(trial.number, epoch + 1))
        
        return max(valROCs) # Return maximum validation set AUC to Optuna 
    
    
if __name__ == "__main__":
    #Add Path to Config - default path will be set to current directory with a config.yaml file
    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group("Input/output options:")
    path_args.add_argument('--config_path', type=str, default="./config.yaml", help='Path to config .yaml file to train/hyperoptimize an InSNPtion model')
    path_args.add_argument("--training_objective", type = str, help = "Training Objective: valid options - {MTL, ST}. MTL implements multi-task learning (Cancer, Ethnicity, and Family History if available). ST implements single-task learning for cancer risk prediction only.")
    args = parser.parse_args()
    
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")
    
    if args.training_objective not in {"ST", "MTL"}:
        raise ValueError("Invalid objective - valid options are: {ST, MTL}.")
    else:
        logger.info("Training task selected: {}".format(args.training_objective))
        
    #Pull model specs from the config:
    config = load_config(args.config_path)
    generalSpecs = config["General"]
        
    #Run Everything based on the Config
    if not generalSpecs["hyperOptimize"]:
        print("Running Single model configuration for {} epochs.".format(generalSpecs["epochs"]))

        trainCancerAcc, trainEthnAcc, trainAUCs, trainSummedLoss, valCancerAcc, valEthnAcc, valAUCs,valSummedLoss = Train_Hyperoptimize(None, config, trainingObjective = args.training_objective)
        
        #Report Results:
        print("Train Accuracies: {} \n".format(trainCancerAcc))
        print("Train Ethnicity Accuracies: {} \n".format(trainEthnAcc))
        print("Train AUCs: {} \n".format(trainAUCs))
        print("Train Losses: {}".format(trainSummedLoss))
        print("\n")
        print("Val Accs: {} \n".format(valCancerAcc))
        print("Val Ethn: {} \n".format(valEthnAcc))
        print("Val AUCs: {} \n".format(valAUCs))
        print("Val losses: {} \n".format(valSummedLoss))
        
        goingOut = pd.DataFrame([trainCancerAcc,trainEthnAcc,trainAUCs,trainSummedLoss, valCancerAcc, valEthnAcc,valAUCs, valSummedLoss]).T
        goingOut.columns = ['Train_PC_Acc', "Train_Ethn_Acc", 'Train_AUC', 'Train_Loss','Val_PC_Acc', 'Val_Ethn_Acc', 'Val_AUC','Val_Loss']
        goingOut.to_csv(os.path.join(generalSpecs['pathForCSVSummary'], generalSpecs["studyName"] + ".tsv"), sep = '\t')       
        
    else:        
        study = optuna.create_study(study_name = generalSpecs["studyName"], direction="maximize", storage = "sqlite:///{}.db".format(os.path.join(generalSpecs["sqlPath"], generalSpecs["studyName"])), load_if_exists = True) 
        logger.info("Using Sampler {}:".format(study.sampler))
        logger.info("Using Pruner {}: \n".format(study.pruner))
        
        try:
            study.optimize(lambda trial: Train_Hyperoptimize(trial, config, trainingObjective = args.training_objective), n_trials=generalSpecs["hyperOptimizeTrials"], timeout=None, n_jobs=generalSpecs["hyperOptimizeJobs"], gc_after_trial = True)
   
        except KeyboardInterrupt:
            logger.warning("Hyperoptimization ended early by user - ending now.../n")
            pass 
        
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        logger.info("Study statistics: ")
        logger.info("  Number of finished trials: ", len(study.trials))
        logger.info("  Number of pruned trials: ", len(pruned_trials))
        logger.info("  Number of complete trials: ", len(complete_trials))

        logger.info("Best trial:")
        trial = study.best_trial

        logger.info("  Value: ", trial.value)

        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info("    {}: {}".format(key, value))
            
        
   
        
