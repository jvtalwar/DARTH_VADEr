'''
@author: James V. Talwar

About: This script loads in the (ideally completed) optimization/search results for baseline ML models (RF, SVM, logreg, XGB), identifies the best performing 
model by defined validation metric (AUC), loads in the corresponding saved model and generates prediction files for each best model type. This script can be 
called by BaselineWinners.sh
'''

import pandas as pd
import optuna 
import os
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from collections import defaultdict
import argparse
from joblib import dump
import yaml 
import sys
from ...src.NeverArgueWithTheDataloader import *
from ...src.AUC_Utils import Calc_ROC
import optuna
import joblib 
import logging
logging.getLogger().setLevel(logging.INFO)


def ExtractTestSet(args):
    #Load in feature set as a SNPDataset object
    featureSet = SNPDataset(feather_file = args.feather_path, pheno_file = args.pheno_path, id_file = args.test_ids, snp_file = args.snp_set, validMafSNPs = args.valid_maf_snps,
                                 id_col = "#IID", ageInclusion = args.age_stats)
    logging.info("First 5 CHR:LOC indexes in feature set: {}".format(featureSet.genotypes.index.tolist()[:5]))
    
    #Load in dataset IDs
    logging.info("Generating Predictions for SNP set {} with a dataset IDs of {}\n".format(args.snp_set.split("/")[-1], args.test_ids.split("/")[-1]))
    testSet = pd.read_csv(args.test_ids, header=None, dtype=str)[0].tolist()
    
    #Extract features, labels, and return
    testFeatures = featureSet.genotypes[testSet]

    if args.include_age:
        logging.info("--include_age specified. Including age as a feature...")
        datasetAges = featureSet.zAge
        testAges = [datasetAges[individual] for individual in testSet]
        testFeatures.loc["zAge"] = testAges

    testLabels = np.array([featureSet.phenotypes[col] for col in testFeatures.columns])
    
    logging.info("Dataset Dimensions: X --> {}   Y --> {}\n".format(testFeatures.T.shape, testLabels.shape))
    
    return testFeatures.T, testLabels


'''
Inputs: 1) np array of features 2) np array of labels 3) args - instantiated command line arguments (or defaults)
Outputs: None - Writes prediction files for the best model (as defined by val metric) for each model type to args.write_path.
'''
def GenerateSubsetPredictions(features, labels, args):
    baselines = {el.split(".")[0]: os.path.join(args.model_summary_path, el) for el in os.listdir(args.model_summary_path) if el.endswith(".db")}
    modelStudy = baselines[args.subset]

    studyResults = optuna.load_study(study_name = args.subset, storage = "sqlite:///{}".format(modelStudy))
    studyResults = studyResults.trials_dataframe()
    studyResults = studyResults[~studyResults.value.isnull()]
    bestTrial = studyResults.loc[studyResults["value"].idxmax(), :] #idxmax returns index of first occurrence of maximum over requested axis

    logging.info("Loading in model {} with best validation AUC of {}".format(os.path.join(args.model_checkpoint_path, "{}_Optuna_Trial_{}.joblib".format(args.subset, bestTrial.number)), bestTrial.value))
    bestModel= joblib.load(os.path.join(args.model_checkpoint_path, "{}_Optuna_Trial_{}.joblib".format(args.subset, bestTrial.number)))
    
    predictionProbabilities = bestModel.predict_proba(features)[:, 1]
    predictionAUC = Calc_ROC(predictionProbabilities, labels) 
    logging.info("Loaded in best {} model has a test set AUC = {}".format(args.subset, predictionAUC))

    predictionDF = pd.DataFrame([predictionProbabilities, labels], index=["Predictions", "Labels"]).T 
    predictionDF.index = pd.read_csv(args.test_ids, header=None, dtype=str)[0].tolist()
    predictionDF.to_csv(os.path.join(args.write_path, "{}_Predictions.tsv".format(args.subset)), sep = "\t")
        
    logging.info("Completed prediction file writing...")
    
    return None

    

'''
Inputs: 1) np array of features 2) np array of labels 3) args - instantiated command line arguments (or defaults)
Outputs: None - Writes prediction files for the best model (as defined by val metric) for each model type to args.write_path.
'''
def GeneratePredictions(features, labels, args):
    hyperoptimizedBaselines = [os.path.join(args.model_summary_path, el) for el in os.listdir(args.model_summary_path) if el.endswith(".db")]
    searchLinearSVM = [os.path.join(args.model_summary_path, el) for el in os.listdir(args.model_summary_path) if el.endswith(".tsv")]
    
    bestModels = dict() #dictionary of best models for each baseline type
    
    for modelStudy in hyperoptimizedBaselines:
        modelType = modelStudy.split("/")[-1].split(".")[0]
        if modelType == "svm": 
            continue
        studyResults = optuna.load_study(study_name = modelType, storage = "sqlite:///{}".format(modelStudy))
        studyResults = studyResults.trials_dataframe()
        studyResults = studyResults[~studyResults.value.isnull()]
        bestTrial = studyResults.loc[studyResults["value"].idxmax(), :] #idxmax returns index of first occurrence of maximum over requested axis
        logging.info("Loading in model {} with best validation AUC of {}".format(os.path.join(args.model_checkpoint_path, "{}_Optuna_Trial_{}.joblib".format(modelType, bestTrial.number)), bestTrial.value))
        bestModels[modelType] = joblib.load(os.path.join(args.model_checkpoint_path, "{}_Optuna_Trial_{}.joblib".format(modelType, bestTrial.number)))
    
    bestReg = float("-inf")
    bestValAuc = float("-inf")
    for c in searchLinearSVM:
        modelSummary = pd.read_csv(c, sep = "\t")
        if modelSummary.loc[0, "ValidationAUC"] > bestValAuc: 
            bestValAuc = modelSummary.loc[0, "ValidationAUC"]
            subString = c.split("/")[-1].split("_")[1].split(".")[0]
            if subString[-2] == "-":
                bestReg = subString[-2:]
            else:
                bestReg = subString[-1]
        
    logging.info("Best linear SVM validation AUC of {} occurs at regularization: {}\n".format(bestValAuc, bestReg))
    bestModels["LinearSVM"] = joblib.load(os.path.join(args.model_checkpoint_path, "LinearSVM_Regularization{}.joblib".format(bestReg)))
    
    #Predict, calculate AUC, and write preds to file
    for modelType, model in bestModels.items():
        if modelType == "LinearSVM":
            predictionProbabilities = model.decision_function(features) 
        else:
            predictionProbabilities = model.predict_proba(features)[:, 1]
            
        predictionAUC = Calc_ROC(predictionProbabilities, labels) 
        logging.info("Loaded in best {} model has a dataset AUC = {}".format(modelType, predictionAUC))
        predictionDF = pd.DataFrame([predictionProbabilities, labels], index=["Predictions", "Labels"]).T 
        predictionDF.index = pd.read_csv(args.test_ids, header=None, dtype=str)[0].tolist()
        predictionDF.to_csv(os.path.join(args.write_path, "{}_Predictions.tsv".format(modelType)), sep = "\t")
        
    logging.info("Completed prediction file writing...")
    
    return None
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group("Input/output options:")
    path_args.add_argument('--snp_set', type = str, required = True, help = 'SNP (feature) set file path used for hyperoptimization.')
    path_args.add_argument('--feather_path', type = str, required = True, help = 'Path to feather file with dataset features.') 
    path_args.add_argument('--pheno_path', type = str, required = True, help = 'Path to feather file with dataset features') 
    path_args.add_argument('--valid_maf_snps', type = str, default = None, help = 'Path to subset of allowable SNPs to be subselected as features.') 
    path_args.add_argument('--test_ids', type = str, required = True, help = 'Path to dataset (e.g., test set) ids file.')     
    path_args.add_argument('--model_summary_path', type = str, required = True, help = "Path where model summaries are found.")
    path_args.add_argument('--model_checkpoint_path', type  = str, required = True, help = 'Path to directory where models are saved.')
    path_args.add_argument('--write_path', type = str, required = True, help = "Path to where prediction files will be written.")
    path_args.add_argument('--include_age', action = "store_true", default = False, help = "Whether to include (z-scored) age as a feature. Default: False.")
    path_args.add_argument("--age_stats", type = str, default = None, help = "Path to cached training set age statistics.")
    path_args.add_argument("--subset", type = str, default = None, help = "Whether to generate predictions for only a certain model type.")

    args = parser.parse_args()
    
    logger = logging.getLogger()
    console = logging.StreamHandler()
    logger.addHandler(console)
    
    useSubset = args.subset in {"logreg", "rf", "xgb"}
    if not useSubset and args.subset is not None:
        logger.warning(f"Invalid/Unsupported subset {args.subset} passed in. Generating predictions for all model baselines.")

    testFeatures, testLabels = ExtractTestSet(args = args)
    
    if useSubset:
        logger.info(f"Generating predictions for model {args.subset}")
        GenerateSubsetPredictions(features = testFeatures, labels = testLabels, args = args)
    else:

        GeneratePredictions(features = testFeatures, labels = testLabels, args = args)                  