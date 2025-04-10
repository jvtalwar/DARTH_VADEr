'''
@author: James V. Talwar

Functions used to generate ROC plots for openSNP test set. If want to save instead of outputting to nb simply pass in save=True to TeamROCket.

Example to run (note first arg==Loaded trained model for first line):

whatAreMyScores = np.array(Meowth(whoeverMadeQualExamsAThingHasEarnedMyEternalIre, test_loader, pleaseBeGpu))
phenotypeLabels = list(pd.read_csv("../Sheen/test_labels.csv").loc[:, "label"])
irisPlexScores = np.array(pd.read_csv("../Sheen/openSNP_final_iris_preds.tsv", sep="\t").loc[:, "brown":"other"])

TeamROCket(whatAreMyScores, phenotypeLabels, "Get Gymwrecked")
TeamROCket(irisPlexScores, phenotypeLabels, "IrisPlex")
'''

import torch
#import seaborn as sns
#import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

def Meowth(model, loader, pleaseBeGpu):
    #print("Hello moto:")
    model.eval()
    predictedScores = list()
    trueLabels = list()
    #print("Things happening")
    #orderOfPheno = will just be pulled from the CSV of labels here: phenotype_file="../Sheen/test_labels.csv", 
    #for i, (snpBatch, phenotypeBatch, ethnBatch) in enumerate(loader): 
    for i, (snpBatch, pcBatch, ethnBatch, fHBatch) in enumerate(loader):
        #print(*phenotypeBatch.tolist())
        snpBatch = snpBatch.to(pleaseBeGpu)
        output = model(snpBatch)
        
        for el in pcBatch.numpy():
            trueLabels.append(el)
        
        for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
            predictedScores.append(el)
     
    return predictedScores,trueLabels

#Generate LD specific predictions and labels - yields a dictionary of clumps instead of tensor snpBatch
def Victreebel(model, loader, pleaseBeGpu):
    model.eval()
    predictedScores = list()
    trueLabels = list()
    for i, (clumpBatch, pcBatch, ethnBatch, fHBatch) in enumerate(loader):    
        gpuClumpBatch = {k:v.to(pleaseBeGpu) for k,v in clumpBatch.items()}
        output = model(gpuClumpBatch)
        
        for el in pcBatch.numpy():
            trueLabels.append(el)
        
        for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
            predictedScores.append(el)
     
    
    return predictedScores,trueLabels


#Generate FC-FFN specific predictions and labels
'''
INPUTS:
    1) Pytorch FC-FFN model (from InSNPtion) 
    2) Dataloader
    3) The device on which Tensor and model operations are performed (i.e., cuda or cpu)
    4) Boolean as to whether age is included or not
    
OUTPUTS:
    1) Model Predictions
    2) Labels (i.e., whether an individual is a case or a control)
'''
def Haunter(model, loader, pleaseBeGpu, ageInclusion):
    model.eval()
    predictedScores = list()
    trueLabels = list()
    
    with torch.no_grad():
        for i, (snpBatch, pcBatch, ethnBatch, fHBatch, zAgeBatch) in enumerate(loader):
            snpBatch = snpBatch.to(pleaseBeGpu)

            #Run it:
            if ageInclusion: #including age - need to pass in more than clump dictionary
                output = model(x = snpBatch, ageBatch = zAgeBatch.to(pleaseBeGpu))
            else:
                output = model(x = snpBatch)

            #Get Labels
            for el in pcBatch.numpy():
                trueLabels.append(el)

            #Get Predictions
            for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
                predictedScores.append(el)
    
    return predictedScores, trueLabels

#Generate JViT/VADEr specific predictions and labels - yields a dictionary of clumps and (z-scored ages!) instead of tensor snpBatch:
def Gengar(model, loader, pleaseBeGpu, ageInclusion):
    model.eval()
    predictedScores = list()
    trueLabels = list()
    
    with torch.no_grad():
        for i, (clumpBatch, pcBatch, ethnBatch, fHBatch, zAgeBatch) in enumerate(loader):    
            gpuClumpBatch = {k:v.to(pleaseBeGpu) for k,v in clumpBatch.items()}

            if ageInclusion: #including age - need to pass in more than clump dictionary
                output = model(dictOfClumps = gpuClumpBatch, ageBatch = zAgeBatch.to(pleaseBeGpu))
            else:
                output = model(dictOfClumps = gpuClumpBatch)

            for el in pcBatch.numpy():
                trueLabels.append(el)

            for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
                predictedScores.append(el)
     
    
    return predictedScores,trueLabels


'''
Inputs: 1) Pytorch model 2) Dataloader 3) Device (e.g., CUDA or cpu) 
        4) Boolean Variable -> Default True; If False includes only age
        information instead of both sex and age (example BCA)
'''
def Persian(model, loader, pleaseBeGpu, allClinical = True):
    model.eval()
    predictedScores = list()
    trueLabels = list()
    
    dtype = torch.FloatTensor
    
    if not allClinical: #Want only Age here
        for i, (snpBatch, pcBatch, ethnBatch, fHBatch, sexBatch, ageBatch) in enumerate(loader):
            snpBatch = snpBatch.to(pleaseBeGpu)
            
            ageBatch = ageBatch.type(dtype) #Ensure correct type for concatenation
            ageBatch = ageBatch.to(pleaseBeGpu)

            output = model(snpBatch, ageBatch)

            for el in pcBatch.numpy():
                trueLabels.append(el)

            for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
                predictedScores.append(el)
    
    else:
        for i, (snpBatch, pcBatch, ethnBatch, fHBatch, sexBatch, ageBatch) in enumerate(loader):
            snpBatch = snpBatch.to(pleaseBeGpu)
            #Inclusion of clinical factors: Ensure correct type for concatenation
            sexBatch = sexBatch.type(dtype) 
            ageBatch = ageBatch.type(dtype)

            #Concatenate clinical features 
            stackedOutput = torch.cat([sexBatch, ageBatch], dim = 1)
            stackedOutput = stackedOutput.to(pleaseBeGpu)    

            #Run it:
            output = model(snpBatch, stackedOutput)

            for el in pcBatch.numpy():
                trueLabels.append(el)

            for el in torch.sigmoid(output[0]).to('cpu').detach().numpy():
                predictedScores.append(el)
        
    
    return predictedScores,trueLabels

#Inputs: #scores, labels, model-name (str), title for ROC generation, save flag, 
def ROCketLauncher(goldenEye,doubleZero,jaqenHagar,title,save = False): 
    lw = 2.3
    fpr,tpr,_ = roc_curve(doubleZero, goldenEye)
    rocketPower = roc_auc_score(doubleZero, goldenEye)
    plt.figure(figsize=(8,8))
    sns.set(font_scale=1.03)
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.plot(fpr,tpr, color = 'red', label = "AUC = {0:0.5f}".format(rocketPower)) #PC ROC curve (area = {0:0.5f})
    plt.xlabel("False Positive Rate", fontsize = 17)
    plt.ylabel("True Positive Rate", fontsize = 17)
    #plt.title("Model " + jaqenHagar.split("/")[-1] + ": Prostate Cancer ROC")
    plt.title("{}".format(title), y = 1.01, fontsize = 19) #Top Breast Cancer InSNPtion ROC:
    plt.legend(loc="lower right", prop={"size":13})
    if save:
        plt.savefig(jaqenHagar + "_ROC.png")
        plt.savefig(jaqenHagar + "_ROC.pdf")

def Calc_ROC(scores, labels):
    rocketPower = roc_auc_score(labels, scores)
    return rocketPower

def TeamROCket(scores, labels, name, save=False):
    y = label_binarize(labels, classes=[0, 1, 2])
    n_classes = y.shape[1]

    y_score = scores

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    plt.figure(figsize=(7,7))
    lw=2.3
    
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='indigo', linestyle=':', linewidth=4)

    colors = cycle(['cyan', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):   
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name + ' Test Set ROC')
    plt.legend(loc="lower right")
    plt.show()
    if save:
        plt.savefig(name + "_ROC.png")
        
    return None