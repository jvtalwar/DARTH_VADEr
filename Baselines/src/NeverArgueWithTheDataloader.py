'''
@author: James V. Talwar

About: Data/dataloader class for genotype (SNP-level) to phenotype prediction through the use of feather objects. Used to generate a dataset class for baseline models and 
a dataloader for FC-FFN (InSNPtion) models.
'''

import numpy as np
import pandas as pd
from pyarrow import feather
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict
import torch
from torch.utils import data
import joblib
import os
from SplitData import DataWriter
import logging
import torch.distributed as dist
logging.getLogger().setLevel(logging.INFO)


#Helper function to order SNPs in order of CHR, LOC for cross-model consistency
def OrderSNPs(genotypes):
    chromosome = list()
    loc = list()
    for snp in genotypes.index:
        splitEmUp = snp.split(":")
        currentChrom = -1

        try:
            currentChrom = int(splitEmUp[0])
        except: 
            if splitEmUp[0].lower() == "x": #chromosome X
                currentChrom = 22
            elif splitEmUp[0].lower() == "y": #chromosome y
                currentChrom = 23
            else:
                raise ValueError("CHR {} does not exist...EXITING".format(splitEmUp[0]))
        
        #assert currentChrom != -1
        
        chromosome.append(currentChrom)
        loc.append(int(splitEmUp[1]))

    genotypes = genotypes.assign(CHR = chromosome, LOC = loc)
    genotypes = genotypes.sort_values(by = ["CHR", "LOC"])
    genotypes = genotypes.drop(["CHR", "LOC"], axis = 1)
    
    return genotypes

class SNPDataset(data.Dataset):
    """Characterizes a dataset for PyTorch.
    Inputs:
    - feather_file - a path to feather file containing genotypes
    - pheno_file - a path to a tsv file with individual phenotypes
    - id_file - a path to the relevant subset of individulas for training, validation, or testing
    - snp_file - a list of SNPs to be optionally subset from genotypes
    - ageInclusion - a string corresponding to the path to training set age statistics. Note this does not pertain to the actual 
                     inclusion of age in the model, but a dictionary of training set age statistics. Age inclusion is handled by
                     model instantiation and training/set set parameters passed in.
    - validMafSNPs - if not None - path to filtered valid SNP subset with a MAF tolerance and consistent genotypes across cohorts 
    - featherWritePath - directory to which write individual feathers (as needed for memory efficiency)
    - id_col - column in pheno_file corresponding to individual ids
    - disease_col - column in pheno_file corresponding to disease labels
    - ancestry_col - column in pheno_file corresponding to ancestries
    - family_his_col - column in pheno_file corresponding to family histories
    """
    def __init__(self, 
                 feather_file, 
                 pheno_file, 
                 id_file, 
                 snp_file,
                 validMafSNPs,
                 featherWritePath = None, 
                 ageInclusion = None,
                 id_col = "IID",
                 disease_col = 'PHENOTYPE', 
                 ancestry_col = 'ANCESTRY', 
                 family_his_col = 'FAMILY_HISTORY'):
            
        # Get relevant metadata about individuals
        pheno = pd.read_csv(pheno_file, sep='\t', dtype={id_col: str})
        self.returnAge = bool(ageInclusion) #whether want data loader to return z-scored ages as well in __getitem__() method
        
        # Create mapping for ancestry to one hot encoded
        anc_categories = [x for x in set([x for x in pheno[ancestry_col].values])]
        labelencoder = LabelEncoder()
        enc = OneHotEncoder(handle_unknown='ignore')
        mp_ancestry = dict(zip(anc_categories, enc.fit_transform(labelencoder.fit_transform(anc_categories).reshape(-1,1)).toarray()))
        self.ancestry = dict(zip(pheno[id_col], pheno[ancestry_col].map(mp_ancestry)))
        
        # Create mapping for family history
        self.fh = dict(zip(pheno[id_col], pheno[family_his_col]))
        
        # Create mapping for 0, 1 label
        if set(pheno[disease_col]) != {0,1}:
            newPheno = pheno[disease_col] - min(pheno[disease_col])
            if set(newPheno) == {0,1}:
                self.phenotypes = dict(zip(pheno[id_col].astype(str), newPheno))
            else:
                raise ValueError("STATUS Values are not 0/1 and cannot be corrected by substraction of min to this representation. Fix phenotype representation.")
        else:
            self.phenotypes = dict(zip(pheno[id_col].astype(str), pheno[disease_col]))
            
        # Get relevant individual ids from list
        self.list_ids = pd.read_csv(id_file, header=None, dtype=str)[0].tolist()
        
        # Get snp ids from list
        self.snp_ids = list(set(pd.read_csv(snp_file, header=None, dtype=str)[0].tolist()))
        invalidSNPPath = "{}".format(os.path.join("/", *snp_file.split("/")[:-1], "InvalidSNPs.txt")) #SNPs with INF or NULL values or train set constant
        invalidSNPs = pd.read_csv(invalidSNPPath, header=None, dtype=str)[0].tolist()  
        
        if len(invalidSNPs) >= 0:
            print("{} SNPs exist across the full 5e-4 dataset with INF or NULL values. Removing these now... Unremoved SNP set size is {}".format(len(invalidSNPs), len(self.snp_ids)))
            self.snp_ids = [el for el in self.snp_ids if el not in invalidSNPs]
            print("Cleaned SNP set size after removal of invalid SNPs is {}".format(len(self.snp_ids)))
        
        
        if validMafSNPs is not None:
            print("Filtering SNP set for MAF and genotype consistent SNPs...")
            consistentMAFGenoSNPs = pd.read_csv(validMafSNPs, header=None, dtype=str)[0].tolist() 
            self.snp_ids = [el for el in self.snp_ids if el in consistentMAFGenoSNPs]
            print("Cleaned SNP set size after filtering incompatible genotype and MAF discrepancy SNPs is {}".format(len(self.snp_ids)))
        
        # Load genotypes
        self.genotypes = feather.read_feather(feather_file).set_index('index')
                    

        validSNPs = sorted(pd.Index(self.snp_ids)[pd.Index(self.snp_ids).isin(self.genotypes.index)])
        self.genotypes = self.genotypes.loc[validSNPs]
        self.genotypes = OrderSNPs(self.genotypes) #Order SNPs by CHR LOC - consistency across modeling approaches
        
        # Writing to files --> only done for FC-FFN networks
        self.RamEfficieny = bool(featherWritePath)
        if self.RamEfficieny:
            # assert path to a directory for feathers exists
            assert os.path.exists(featherWritePath)

            logging.info("Valid featherWritePath {} passed in. Implementing data writer to reduce RAM requirements for dataloading.".format(featherWritePath))
            self.genotypeFolderDirectory = featherWritePath
            dataWriter = DataWriter(genotypes = self.genotypes, idToFilenamePath = featherWritePath) 
            dataWriter.splitData() #at some point this should be moved earlier in instance files are already written so can skip the whole genotype preprocessing...
            
            del self.genotypes
            
        if bool(ageInclusion): #Age inclusion is not None
            assert os.path.exists(ageInclusion) #assert that it is a valid path
            
            ageStats = joblib.load(ageInclusion)
            
            #add a z-age column bounded to the training set age range to phenos dataframe and add a z-scored age structure that can be pulled from in __getitem__ 
            pheno["zAge"] = pheno["AGE"].apply(lambda x: (max(min(x, ageStats["maxAge"]), ageStats["minAge"]) - ageStats["mu"])/ageStats["sigma"]) 
            pheno.zAge.fillna(0, inplace = True) #if there are individuals with None/NaNs for age fill them with the z-scored mean (i.e., zero)
            self.zAge = dict(zip(pheno[id_col], pheno["zAge"]))
            logging.info("Valid age path given: {}\n   Returning z-scored ages in loader...".format(ageInclusion))
        
    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        """Returns one data pair (genotypes and label)."""
        
        # Grab id of individual
        ID = self.list_ids[index]
        
        # Grab SNP data
        if self.RamEfficieny:
            genotype = feather.read_feather(os.path.join(self.genotypeFolderDirectory, "genotype_data_record_{}.feather".format(ID)))
            genotype = genotype[ID].values
            
        else:
            genotype = self.genotypes[ID].values
        
        X = torch.from_numpy(genotype).float()
        
        # Grab cancer diagnosis
        cancer = torch.from_numpy(np.array([self.phenotypes[ID]]))
        
        # Grab ancestry value    
        z = torch.from_numpy(self.ancestry[ID])
        anc = torch.Tensor([torch.argmax(z)])
        
        # Grab family history
        fh = torch.from_numpy(np.array([self.fh[ID]]))
        
        if self.returnAge:
            zAge = torch.from_numpy(np.array([self.zAge[ID]])).float()
            return X, cancer, anc, fh, zAge
        
        # Yield it
        return X, cancer, anc, fh 


def get_loader(feather, phenos, ids, snps, id_column, batch_size, shuffle, num_workers, validMafSnpPath, featherWritePath = None, ageTrainStats = None):
    """Returns torch.utils.data.DataLoader for geno dataset."""
    
    snp_dataset = SNPDataset(feather_file = feather, pheno_file = phenos, id_file = ids, snp_file = snps, id_col = id_column, ageInclusion = ageTrainStats, validMafSNPs = validMafSnpPath, featherWritePath = featherWritePath)    
    params = {'batch_size': batch_size, 'shuffle': shuffle,'num_workers': num_workers}
    data_loader = torch.utils.data.DataLoader(dataset = snp_dataset, **params)
    
    return data_loader
    
    