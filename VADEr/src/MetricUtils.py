'''
@author: James V. Talwar
Created on 2/29/2024 at 19:46:53

About: MetricUtils.py contains metric functions for evaluating VADEr and FC-FFN model performances, including ROC AUC and accuracy (both binary and muti-class predictions)
'''

import torch
import torch.nn.functional as F
import logging
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


'''
About: Method to calculate accuracy for binary class prediction for models with unactivated (i.e., non-sigmoid probability transformed) outputs.


Previously: Accuracy here is not a probability of correctness, but instead is the total number of correct predictions, with probability conversion via the total 
number of labels/predictions handled in training/evaluation calls (done to allow easy distributed training and tracking - can use reduce) 


Inputs: 1) preds: torch tensor of unactivated model predictions 
        2) labels: torch tensor of labels (0/1)
Output(s): correctNumPreds: An integer corresponding to the total number of correct predictions. 
'''
def BinaryClassAccuracy(preds, labels):
    predictions = torch.round(torch.sigmoid(preds))
    return sum(predictions.eq(labels)).item()/labels.shape[0]

'''
About: Method to calculate accuracy for multiclass class prediction for models with unactivated (i.e., non-softmax probability transformed) outputs.


Previously: Accuracy here is not a probability of correctness, but instead is the total number of correct predictions, with probability conversion via the total 
number of labels/predictions handled in training/evaluation calls (done to allow easy distributed training and tracking - can use reduce) 


Inputs: 1) preds: torch tensor of unactivated model predictions 
        2) labels: torch tensor of labels with the numerical value representing the target class of the label
Output(s): correctNumPreds: An integer corresponding to the total number of correct predictions. 
'''
def MultiClassAccuracy(preds, labels):
    predictions = torch.argmax(F.softmax(preds, dim = 1), dim = 1).unsqueeze(1)
    return sum(predictions.eq(labels)).item()/labels.shape[0]

'''
About: Method to calculate ROC AUC for a model's predictions. Employs
sklearn's roc_auc_score.

Input(s): 1) preds: numpy array of model predictions 
          2) labels: numpy array of true labels 
Output(s): Float corresponding to model ROC AUC
'''
def Calc_ROC_AUC(preds, labels):
    return roc_auc_score(labels, preds) #expects labels first, predictions second