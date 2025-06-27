'''
@author: James V. Talwar

About: Utility functions to help calculate AUC scores for baseline models.
unctions used to generate ROC plots for openSNP test set. If want to save instead of outputting to nb simply pass in save=True to TeamROCket.
'''

import torch
from sklearn.metrics import roc_auc_score

'''
About: Generate FC-FFN model specific predictions and labels
INPUTS:
    1) Pytorch FC-FFN model (from InSNPtion) 
    2) Dataloader
    3) The device on which Tensor and model operations are performed (i.e., cuda or cpu)
    4) Boolean as to whether age is included or not
    
OUTPUTS:
    1) Model Predictions
    2) Labels (i.e., whether an individual is a case or a control)
'''
def Generate_FFN_Preds(model, loader, device, ageInclusion):
    model.eval()
    predictedScores = list()
    trueLabels = list()
    
    with torch.no_grad():
        for i, (snpBatch, pcBatch, ethnBatch, fHBatch, zAgeBatch) in enumerate(loader):
            snpBatch = snpBatch.to(device)

            #Run it:
            if ageInclusion: #including age - need to pass in more than clump dictionary
                output = model(x = snpBatch, ageBatch = zAgeBatch.to(device))
            else:
                output = model(x = snpBatch)

            #Get Labels
            for el in pcBatch.numpy():
                trueLabels.append(el)

            #Get Predictions
            for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
                predictedScores.append(el)
    
    return predictedScores, trueLabels

def Calc_ROC(scores, labels):
    rocketPower = roc_auc_score(labels, scores)
    return rocketPower
