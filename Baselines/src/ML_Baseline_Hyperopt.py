'''
@author: James V. Talwar
Created on October 25, 2022 at 19:56:41

About: This script performs baseline hyperoptimization for comparison against VADEr for complex disease prediction. 
Baseline models hyperoptimized here include logistic regression, xgb, random forests, and a linear kernel SVM. A general 
SVM (kernel = linear, poly, rbf) can also be be hyperoptimized as well, but is extremely time intensive (and for 
benchmarking purposed we used a linear kernel SVM). For more details see below note.

NOTE:
- For the linear kernel SVM, for time/computational efficiency, LinearSVC is the method used for fitting, excepting overflow errors 
that can occur with liblinear for high dimensional feature sets. In such an instance a SVM approximation/equivalency is used
employing linear_model.SGDClassifier. See below comments for details. 
'''


import pandas as pd
import time
import optuna 
import os
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
import numpy as np
import sys
from collections import defaultdict
import argparse
from joblib import dump
import yaml 
import sys
from NeverArgueWithTheDataloader import *  
from AUC_Utils import Calc_ROC
import optuna
import joblib
import logging
logging.getLogger().setLevel(logging.INFO)


def ExtractTrainAndVal(args):
    #Load in feature set as a SNPDataset object
    featureSet = SNPDataset(feather_file = args.feather_path, pheno_file = args.pheno_path, id_file = args.train_ids, snp_file = args.snp_set, validMafSNPs = args.valid_maf_snps,
                                 id_col = "#IID", ageInclusion = args.age_stats) 
    
    logging.info("First 5 CHR:LOC indexes in feature set: {}".format(featureSet.genotypes.index.tolist()[:5]))
    
    #Load in train and validation IDs
    trainSet = pd.read_csv(args.train_ids, header=None, dtype=str)[0].tolist()
    valSet = pd.read_csv(args.val_ids, header=None, dtype=str)[0].tolist()


    #Extract train and val features, labels, and return
    trainFeatures = featureSet.genotypes[trainSet]

    if args.include_age:
        logging.info("--include_age specified. Including age as a feature...")
        datasetAges = featureSet.zAge
        trainAges = [datasetAges[individual] for individual in trainSet]
        valAges = [datasetAges[individual] for individual in valSet]
        
        trainFeatures.loc["zAge"] = trainAges
        

    trainLabels = np.array([featureSet.phenotypes[col] for col in trainFeatures.columns])
    logging.info("\nTrain Dimensions: X --> {}   Y --> {}".format(trainFeatures.T.shape, trainLabels.shape))

    valFeatures = featureSet.genotypes[valSet]
    if args.include_age:
        valFeatures.loc["zAge"] = valAges

    valLabels = np.array([featureSet.phenotypes[col] for col in valFeatures.columns])
    
    logging.info("Val Dimensions: X --> {}   Y --> {}\n".format(valFeatures.T.shape, valLabels.shape))
    
    return trainFeatures.T, trainLabels, valFeatures.T, valLabels

'''
Handle Linear SVM separately as a simple C parameter search over regularization values
Inputs: 1) Train X (np.array) 2) Train Y 3) Validation X 4) Validation Y (welcome to Gerudo Valley Link!) 5) Regularization parameter (int) 
6) Model checkpoint path (str) 7) Path to save model results.

NOTE: Using LinearSVC for speed of convergence with high number of points here. SVC is v slow for high number of points/feature size - trying linear SVC which uses liblinear and thus scales better 
(though is different optimization metric) and setting loss = "hinge" (as opposed to default "squared_hinge") as that is what is used by SVC. See: https://stackoverflow.com/questions/35076586/when-should-one-use-linearsvc-or-svc
'''
def LinearSVMRegularization(trainX, trainY, valX, valY, c, modelCheckpointPath, fileName = "./LinearSVMOptimization.tsv"):   
    finalStats = defaultdict(lambda: defaultdict(float))
    
    logging.info("Train feature set size is {} rows (samples) and {} columns (features). Using Dual ==  {}".format(trainX.shape[0], trainX.shape[1], (trainX.shape[0] < trainX.shape[1])))
    
    try:
        model = LinearSVC(loss = "hinge", C = (10**c), random_state = 3) #Dual parameter should theoretically be adapted based on SNP set size. From documentation: Prefer dual=False when n_samples > n_features. Default = true. However when using: dual = (trainX.shape[0] < trainX.shape[1]) liblinear returns a value error saying hinge and l2 loss with dual=False is not supported, so leaving as default.
        model.fit(trainX, trainY)
    
    except:
        logging.warning("Liblinear likely threw an Overflow error (unless issues elsewhere - e.g., feature set dimensions, values, scale, etc.) using SGDClassifier with hinge loss for SVM...")
        model = linear_model.SGDClassifier(loss='hinge', alpha = (10**c), random_state = 3) #liblinear (solver for LinearSVC) for some reason fails w/ OverflowError: value too large to convert to npy_int32 for largest SNP set: 5e-4. Unclear reasoning why as no features are inf or nan so is likely due to feature set size (especially since log reg and rf both work). SGDClassifier with "hinge" loss gives a linear SVM. The default loss is equivalent to l2 which matches LinearSVC. Regularization parameter here is alpha which is not equivalent to C - but reg search should theoretically find the best one if inversely proportional. https://stats.stackexchange.com/questions/216095/how-does-alpha-relate-to-c-in-scikit-learns-sgdclassifier. Expect largest alpha to perform best for this...
        model.fit(trainX, trainY)

    trainPredictions = model.predict(trainX)
    valPredictions = model.predict(valX)
    trainProbs = model.decision_function(trainX) #SVM so using decision function (i.e., where and how far a sample lies from the hyperplane). Essentially gives a SVM score
    valProbs = model.decision_function(valX) 

    trainAcc = sum(trainPredictions == trainY)/len(trainY)
    valAcc = sum(valPredictions == valY)/len(valY)

    trainAUC = Calc_ROC(trainProbs, trainY) #
    valAUC = Calc_ROC(valProbs, valY)

    trainPR_AUC = average_precision_score(trainY, trainProbs)
    valPR_AUC = average_precision_score(valY, valProbs)

    finalStats[c]["TrainAccuracy"] = trainAcc
    finalStats[c]["ValidationAccuracy"] = valAcc
    finalStats[c]["TrainAUC"] = trainAUC
    finalStats[c]["ValidationAUC"] = valAUC
    finalStats[c]["TrainPR_AUC"] = trainPR_AUC
    finalStats[c]["ValidationPR_AUC"] = valPR_AUC

    
    results = pd.DataFrame(finalStats).T
    results.to_csv(fileName, sep = "\t", index = False)
    
    #Save model
    dump(model, os.path.join(modelCheckpointPath, "LinearSVM_Regularization{}.joblib".format(c)))
    
    return None

'''
Hyperoptimization Method Called by Optuna:

Inputs:
1) trial: Optuna Trial
2) trainX: training set features (i.e., genotypes)
3) trainY: training set labels (i.e., phenotypes)
4) valX: validation set features
5) valY: validation set labels
6) modelCheckpointPath: path to save model
7) method: model type (str) - can be one of {logreg, rf, svm, xgb}
8) num_gpus: number of GPUs to use for xgboost (int)
'''
def DoTheUhThing(trial, trainX, trainY, valX, valY, modelCheckpointPath, method = "logreg", num_gpus = 3):
    method = method.lower()
    validOptions = {"logreg", "rf", "svm", "xgb"}
    
    assert method in validOptions, f"Method method = {method} not recognized. Valid method inputs are: logreg, svm, xgb, rf."
    
    if method == "logreg":
        penalty = trial.suggest_categorical("LogReg_penaltyfx", ["l1", "l2", "elasticnet"]) 
        
        #select solver based on selected regularization
        if penalty == "l1":
            solveAnAlgebraicEqnInTheBlinkOfABeesEye = "saga" #"liblinear"
        elif penalty == "elasticnet":
            solveAnAlgebraicEqnInTheBlinkOfABeesEye = "saga"
        else:
            solveAnAlgebraicEqnInTheBlinkOfABeesEye = "lbfgs" #the default 
        
        c = trial.suggest_int("LogReg_Regularization", -5,5) #Inverse of regularization strength; must be a positive float. Like SVMs, smaller values specify stronger regularization. 
        
        #partition by elastic net to avoid err file spitting out endless warnings...
        if penalty == "elasticnet":
            elasticNetRatio = trial.suggest_float("LogReg_ElasticNetL1Ratio", 0, 1)
            model = linear_model.LogisticRegression(penalty = penalty, solver = solveAnAlgebraicEqnInTheBlinkOfABeesEye,
                                                C=(10**c), l1_ratio = elasticNetRatio, max_iter = 1000, random_state = 3)
        else:
            model = linear_model.LogisticRegression(penalty = penalty, solver = solveAnAlgebraicEqnInTheBlinkOfABeesEye,
                                                C=(10**c), max_iter = 1000, random_state = 3)
        
    elif method == "rf":
        numTrees = trial.suggest_int("RF_n_estimators", 50, 500) 
        maxDepth = trial.suggest_int("rf_max_depth", 2, 7) 
                
        #maxFeatures = trial.suggest("RF_max_features", ) --> leaving this to default auto which is = sqrt(n_features)
        
        minSamplesSplit = trial.suggest_int("RF_min_samples_split", 2, 50) #2,10; minimum num samples to split an internal node; 2 is default value --> should f(x) as regularization at higher numbers
        minSamplesLeaf = trial.suggest_int("RF_min_samples_leaf", 1, 10) #1,5; default is 1 --> higher numbers correspond to a regularization - i.e. can't keep expanding tree depth until all leaves are pure
        strapInItsGoingToBeABumpyRide = trial.suggest_categorical("RF_bootstrap", [True, False]) #default is True
        model = RandomForestClassifier(n_estimators=numTrees, max_depth = maxDepth, bootstrap=strapInItsGoingToBeABumpyRide, 
                                       min_samples_split=minSamplesSplit, min_samples_leaf=minSamplesLeaf, random_state = 3)
    
    elif method == "svm":
        c = trial.suggest_int("SVM_Regularization", -5,5)
        kernel = trial.suggest_categorical("SVM_kernel", ["linear", "poly", "rbf"])
        if kernel == "poly":
            degree = trial.suggest_int("SVM_Degree", 2,5) #only for polynomial functions
            
        if kernel in {"poly", "rbf"}: #gamma is not a param for linear
            #Side note: Gamma affects the variance (likelihood of overfitting) - low gamma = higher variance vs. high gamma = lower variance 
            gamma = trial.suggest_categorical("SVM_GammaMethod", ["scale", "auto", "number"]) #default is scale == 1 / (n_features * X.var())
            if gamma == "number":
                gamma = trial.suggest_float("SVM_Gamma_Number",0.001,100)
                
        if kernel == "linear":
            model = SVC(kernel=kernel,probability=True, C = (10**c), random_state = 3) #enable probabilty estimates
        elif kernel == "poly":
            model = SVC(kernel=kernel,probability=True, C = (10**c), gamma = gamma, degree = degree, random_state = 3) #add degree and gamma for polynomial
        else: #rbf
            model = SVC(kernel=kernel, probability=True, C = (10**c), gamma = gamma, random_state = 3)
    
    elif method == 'xgb':
        learning_rate = trial.suggest_float("eta", 1e-2, 3e-1, log = True) #Increase likelihood of sampling from smaller values
        n_estimators = trial.suggest_int("num_boosting_rounds", 100, 3000, step = 100)
        max_depth = trial.suggest_int("max_depth", 2, 7)
        gamma = trial.suggest_int("gamma", 0, 5) #Cost which must be improved upon to split a node
        alpha = trial.suggest_int("alpha", 0, 10) #L1
        lamb = trial.suggest_int("lambda", 0, 10) #L2
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10) #sum of the instance weights which must be exceeded to continue building - larger == more regularization
        subsample = trial.suggest_float("subsample", 0.3, 1, step = 0.1) #helps prevent overfitting
        colsample_bytree = trial.suggest_float("subsample", 0.3, 1, step = 0.1) #subsample of features to use for a tree - XGB dropout essentially

        xgb_params = {'eta': learning_rate,
                      'n_estimators': n_estimators,
                      'max_depth': max_depth, 
                      'alpha': alpha,
                      'gamma': gamma, 
                      'lambda': lamb,
                      'min_child_weight': min_child_weight,
                      'subsample': subsample,
                      'colsample_bytree': colsample_bytree,
                      'eval_metric': 'auc',
                      'early_stopping_rounds': 50}
        
        model = XGBClassifier(**xgb_params, tree_method = "gpu_hist", gpu_id = (trial.number % num_gpus), random_state = 3) 
        
    else: 
        raise ValueError("Invalid Method...")
        
    #Fit model and get val AUC
    if method == "xgb":
        model.fit(trainX, trainY, eval_set = [(valX, valY)])
    else:
        model.fit(trainX, trainY)

    valProbs = model.predict_proba(valX)[:, 1]
    validationAUC = Calc_ROC(valProbs, valY)  
    
    #Save model so don't need to retrain
    dump(model, os.path.join(modelCheckpointPath, "{}_Optuna_Trial_{}.joblib".format(method, trial.number)))
    
    return validationAUC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group("Input/output options:")
    path_args.add_argument('--snp_set', type = str, help = 'SNP (feature) set file path used for hyperoptimization.')
    
    path_args.add_argument('--feather_path', type = str, help = 'Path to feather file with dataset features.') 
    path_args.add_argument('--pheno_path', type = str, help = 'Path to feather file with dataset features') 
    path_args.add_argument('--valid_maf_snps', type = str, default = None , help = 'Path to subset of allowable SNPs to be subselected as features. Assumption is cross-cohort compatible MAF SNPs (but None is a viable parameter if want all features).') 
    path_args.add_argument('--train_ids', type = str, help = 'Path to training ids file.')
    path_args.add_argument('--val_ids', type = str, help = 'Path to validation ids file.') 
        
    path_args.add_argument('--model_type', type = str, default = "logreg", help = 'Model type for hyperoptimization. Can select from {xgb, rf, logreg, svm_linear, svm} .')
    path_args.add_argument('--num_trials', type = int, default = 100, help = 'Number of trials want to run per process/thread/CPU. Essentially the number of trials per independent call to script. This number is irrelevant for model_type == svm_linear, which employs regularization grid search.')
    path_args.add_argument('--rdb_path', type = str, help = "Path to write optuna study.")
    path_args.add_argument('--model_checkpoint_path', type  = str, help = 'Path to directory where want to save models.')
    path_args.add_argument('--regularization', type = int, help = "When if --model_type is set to svm_linear provide a regularization parameter. This parameter is c in C = 10**c. Suggested range: [-5, 5].")
    path_args.add_argument('--include_age', action = "store_true", default = False, help = "Whether to include (z-scored) age as a feature. Default: False.")
    path_args.add_argument("--age_stats", type = str, default = None, help = "Path to cached training set age statistics.")

    args = parser.parse_args()
    
    logger = logging.getLogger()
    console = logging.StreamHandler()
    logger.addHandler(console)
       
    if args.valid_maf_snps == "None":
        args.valid_maf_snps = None
    
    trainX, trainY, valX, valY = ExtractTrainAndVal(args)
    
    logging.info("Instantiated... running hyperopt for model type {}".format(args.model_type))
    start = time.time()
    
    if args.model_type.lower() == "svm_linear": #linear svm has independent function that searches C parameters
        LinearSVMRegularization(trainX, trainY, valX, valY, c = args.regularization, modelCheckpointPath = args.model_checkpoint_path, fileName = os.path.join(args.rdb_path, "LinearSVMOptimization_Regularization{}.tsv".format(args.regularization)))
        end = time.time()
        logging.info("Elapsed time after instantiation for model type {}: {}".format(args.model_type, end - start))
        
    else: #optuna hyperparameter search
        study = optuna.create_study(study_name = args.model_type, direction = "maximize", storage = "sqlite:///{}.db".format(os.path.join(args.rdb_path, args.model_type)), load_if_exists = True) 
        
        try:
            study.optimize(lambda trial: DoTheUhThing(trial, trainX, trainY, valX, valY, args.model_checkpoint_path, method = args.model_type), n_trials = args.num_trials, timeout = None, n_jobs = 4) 
            
        except KeyboardInterrupt:
            logging.warning("Hyperoptimization ended early by user - ending now.../n")
            pass 
        
        end = time.time()
        logging.info("Elapsed time after instantiation for model type {}: {}".format(args.model_type, end - start))
        
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        logging.info("Study statistics: ")
        logging.info("  Number of finished trials: {}".format(len(study.trials)))
        logging.info("  Number of pruned trials: {}".format(len(pruned_trials)))
        logging.info("  Number of complete trials: {}".format(len(complete_trials)))

        logging.info("Best trial:")
        trial = study.best_trial
        logging.info("  Value: {}".format(trial.value))

        logging.info("  Params: ")
        for key, value in trial.params.items():
            logging.info("    {}: {}".format(key, value))
    