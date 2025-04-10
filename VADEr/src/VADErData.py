'''
@author: James V. Talwar

About: VADErData.py defines a SNP_Dataset class that can be used for both distributed and non-distributed VADEr training. It also 
contains code for preprocessing and caching full-dataset feathers for memory efficiency.
'''

import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import joblib
from pyarrow import feather
import tqdm
from VADErDataUtils import LoadAndProcessGenotypes, CachemAllPatchymon, GeneratePatches
import logging

logger = logging.getLogger(__name__)
#logging.getLogger().setLevel(logging.INFO)

class SNP_Dataset(Dataset):
    def __init__(self,
                 featherFilePath,
                 phenoFilePath,
                 idFilePath, 
                 snpSubsetPath,
                 validMafSnpsPath,
                 vaderPatchMappingPath,
                 enableShifting, #Boolean corresponding to whether to load shifted patches (i.e., using SPT)
                 cacheWritePath = None,
                 cacheStorageMethod = "feather",
                 trainingSetAgeStatsPath = None,
                 sparsePatchThreshold = 0, #The number to be excluded should be 1 less than this parameter (all patches of density >= sparsePatchThreshold will be kept). Default 0, but implemented to allow user to omit sparse patches.
                 idCol = "#IID",
                 diseaseCol = 'PHENOTYPE', 
                 ancestryCol = 'ANCESTRY', 
                 familyHistoryCol = 'FAMILY_HISTORY'):
        
        # Handle case where sparsePatchThreshold is not specified in config (i.e., returns None)
        if sparsePatchThreshold is None:
            sparsePatchThreshold = 0

        # Get relevant metadata about individuals
        pheno = pd.read_csv(phenoFilePath, sep = '\t', dtype = {idCol: str})
        self.returnAge = bool(trainingSetAgeStatsPath) #whether want data loader to return z-scored ages as well in __getitem__() method
        
        # Create mapping for ancestry to one hot encoded
        ancestryCategories = [x for x in set([x for x in pheno[ancestryCol].values])]
        labelEncoder = LabelEncoder()
        ohEncoder = OneHotEncoder(handle_unknown = 'ignore')
        ancestryOneHotMapping = dict(zip(ancestryCategories, ohEncoder.fit_transform(labelEncoder.fit_transform(ancestryCategories).reshape(-1,1)).toarray()))
        self.ancestry = dict(zip(pheno["#IID"], pheno["ANCESTRY"].map(ancestryOneHotMapping)))
        
        # Create mapping for family history
        self.fh = dict(zip(pheno[idCol], pheno[familyHistoryCol]))
        
        # Create mapping for 0, 1 label
        if set(pheno[diseaseCol]) != {0,1}:
            newPheno = pheno[diseaseCol] - min(pheno[diseaseCol])
            if set(newPheno) == {0,1}:
                self.phenotypes = dict(zip(pheno[idCol].astype(str), newPheno))
            else:
                raise ValueError("STATUS Values are not 0/1 and cannot be corrected by substraction of min to this representation. Fix phenotype representation.")
        else:
            self.phenotypes = dict(zip(pheno[idCol].astype(str), pheno[diseaseCol]))
            
        # Get relevant individual ids from list
        self.datasetIDs = pd.read_csv(idFilePath, header = None, dtype = str)[0].tolist()
        
        if self.returnAge: #Age inclusion is not None
            assert os.path.exists(trainingSetAgeStatsPath), f"trainingSetAgeStatsPath at {trainingSetAgeStatsPath} does not exist. EXITING..." #assert that it is a valid path
            
            ageStats = joblib.load(trainingSetAgeStatsPath)
            
            #add a z-age column bounded to the training set age range to phenos dataframe and add a z-scored age structure that can be pulled from in __getitem__ 
            pheno["zAge"] = pheno["AGE"].apply(lambda x: (max(min(x, ageStats["maxAge"]), ageStats["minAge"]) - ageStats["mu"])/ageStats["sigma"]) 
            pheno.zAge.fillna(0, inplace = True) #if there are individuals with None/NaNs for age fill them with the z-scored mean (i.e., zero)
            self.zAge = dict(zip(pheno[idCol], pheno["zAge"]))
            logger.info("Valid age path given: {}\n   Returning z-scored ages in loader...".format(trainingSetAgeStatsPath))
        
        self.RAM_Efficiency = bool(cacheWritePath)
        self.enableShifting = enableShifting

        if self.RAM_Efficiency:
            # assert path to a directory for individual files exists
            assert os.path.isdir(cacheWritePath), f"Provided cacheWritePath {cacheWritePath} is not a directory/folder. EXITING..."
            assert cacheStorageMethod in {"feather", "joblib"}, f"Provided cacheStorageMethod {cacheStorageMethod} invalid. cacheStorgae method allowable options: 'feather', 'joblib'"
            
            logger.info(f"Caching with storage method {cacheStorageMethod}")
            preprocessed = os.path.exists(os.path.join(cacheWritePath, "featurePatchSizes.joblib"))
            
            if not preprocessed:
                genotypes = LoadAndProcessGenotypes(snpSubsetPath = snpSubsetPath,
                                                    validMafSnpsPath = validMafSnpsPath,
                                                    featherFilePath = featherFilePath)

                #cache/further preprocess data to individual file level...
                CachemAllPatchymon(genotypes = genotypes, 
                                   patchMappingPath = vaderPatchMappingPath, 
                                   writePath = cacheWritePath, 
                                   datasetIDs = self.datasetIDs,
                                   sparsityThreshold = sparsePatchThreshold,
                                   enableShifting = enableShifting,
                                   cacheMethod = cacheStorageMethod)
            
            self.patchSizes = joblib.load(os.path.join(cacheWritePath, "featurePatchSizes.joblib"))
            self.featurePatchToSNPs = joblib.load(os.path.join(cacheWritePath, "featurePatchToSNPs.joblib"))
            self.cachedPath = cacheWritePath
            self.sparsityThreshold = sparsePatchThreshold
            self.cacheStorageMethod = cacheStorageMethod

        else:
            genotypes = LoadAndProcessGenotypes(snpSubsetPath = snpSubsetPath,
                                                validMafSnpsPath = validMafSnpsPath,
                                                featherFilePath = featherFilePath)
            
            genotypes = GeneratePatches(genotypes = genotypes, patchMappingPath = vaderPatchMappingPath) #Add patch column to genotypes and omit sparse patches from feature mapping dictionaries 
            groupedGenotypes = genotypes.groupby("Patch", observed = True)
            
            self.genotypeDict = dict() #mapping of each patch to full dataset subset of genotypes for that patch - markedly improves __getitem__ speed compared to conversion of grouped DF group genos 
            self.patchSizes = dict()
            self.featurePatchToSNPs = dict() #Added in case want access down the line
            
            
            for group, data in tqdm.tqdm(groupedGenotypes):
                numSNPsInPatch = data.shape[0]
                if numSNPsInPatch >= sparsePatchThreshold: #In case want to omit certain sparse patches (default is 0) so all patches of 1 or more SNPs will be included
                    self.patchSizes[group] = numSNPsInPatch
                    self.featurePatchToSNPs[group] = data.index.tolist() 
                    self.genotypeDict[group] = data

            if enableShifting:
                #Note - groupby drops NaNs by default
                leftShiftedGenos = genotypes.groupby("Left_Shifted_Patch", observed = True)
                rightShiftedGenos = genotypes.groupby("Right_Shifted_Patch", observed = True)

                #concatenate left shifted and right shifted patches with current results
                patches = [patch for patch in self.genotypeDict.keys()]
                
                #NOTE: Doesn't account for whether a left or right shifted patch incorporates from a potentially skipped patch due to below sparsity threshold
                for patch in patches:
                    if patch in leftShiftedGenos.groups.keys():
                        self.genotypeDict[patch] = pd.concat([self.genotypeDict[patch], leftShiftedGenos.get_group(patch)], axis = 0)
                    
                    if patch in rightShiftedGenos.groups.keys():
                        self.genotypeDict[patch] = pd.concat([self.genotypeDict[patch], rightShiftedGenos.get_group(patch)], axis = 0)

                    self.patchSizes[patch] = self.genotypeDict[patch].shape[0]
                    self.featurePatchToSNPs[patch] = self.genotypeDict[patch].index.tolist()


            #self.genotypes, self.patchSizes = GroupedPatchesToDictionary(groupedDF = groupedGenotypes, datasetIDs = self.datasetIDs)

        
    def __len__(self):
        return len(self.datasetIDs)
    
    def __getitem__(self, index):
        ID = self.datasetIDs[index] #extract ID of individual

        #Extract genotype-patch information of individual (depending on whether caching specified)
        if self.RAM_Efficiency:
            if (not self.enableShifting) and (self.cacheStorageMethod == "feather"):
                groupedGeno = feather.read_feather(os.path.join(self.cachedPath, f"{ID}.feather")).groupby("Patch")
                featureDict = {group: torch.from_numpy(data[ID].values).float() for group, data in groupedGeno if data.shape[0] >= self.sparsityThreshold}
            else:
                featureDict = joblib.load(os.path.join(self.cachedPath, f"{ID}.joblib"))
            
            #geno = feather.read_feather(os.path.join(self.cachedPath, f"{ID}.feather"))
            #featureDict = {patch: torch.from_numpy(geno.loc[snps, ID].values).float() for patch, snps in self.featurePatchToSNPs.items()}
            
            #groupedGeno = feather.read_feather(os.path.join(self.cachedPath, f"{ID}.feather")).groupby("Patch")
            #featureDict = {group: torch.from_numpy(data[ID].values).float() for group, data in groupedGeno if data.shape[0] >= self.sparsityThreshold} #Handle sparsity thresholding 

            #featureDict = joblib.load(os.path.join(self.cachedPath, "{ID}.joblib"))
        
        else:
            featureDict = dict()
            for k,v in self.genotypeDict.items():
                featureDict[k] = torch.from_numpy(v[ID].values).float()

            #featureDict =  {group: torch.from_numpy(data[ID].values).float() for group, data in self.groupedGenotypes} #self.groupedGenotypes[ID]

        # Get labels
        diseaseStatus = torch.from_numpy(np.array([self.phenotypes[ID]]))

        # ancestry 
        z = torch.from_numpy(self.ancestry[ID])
        ancestry = torch.Tensor([torch.argmax(z)])

        # family history (in case desired) 
        famHis = torch.from_numpy(np.array([self.fh[ID]]))

        # Return all values
        if self.returnAge:
            zScoredAges = torch.from_numpy(np.array([self.zAge[ID]])).float()  
            return featureDict, diseaseStatus, ancestry, famHis, zScoredAges
        
        return featureDict, diseaseStatus, ancestry, famHis