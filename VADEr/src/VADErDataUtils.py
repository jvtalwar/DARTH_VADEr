'''
@author: James V. Talwar

About: All necessary utility functions for processing input data and creating a SNP_Dataset object. Utilized by VADErData.py.
'''

import pandas as pd
import os
from pyarrow import feather
import joblib
from joblib import Parallel, delayed
import torch
import torch.nn.functional as F
import numpy as np
import logging
import tqdm

logger = logging.getLogger(__name__)

'''
About: Helper function to order SNPs in order of CHR, LOC for cross-model consistency
'''
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
                currentChrom = 23
            elif splitEmUp[0].lower() == "y": #chromosome y
                currentChrom = 24
            else:
                raise ValueError("CHR {} does not exist...EXITING".format(splitEmUp[0]))
        
        #assert currentChrom != -1
        
        chromosome.append(currentChrom)
        loc.append(int(splitEmUp[1]))

    genotypes = genotypes.assign(CHR = chromosome, LOC = loc)
    genotypes = genotypes.sort_values(by = ["CHR", "LOC"])
    genotypes = genotypes.drop(["CHR", "LOC"], axis = 1)
    
    return genotypes

'''
About: Method to load, preprocess, and consistently index a full-dataset's genotypes.

Input(s): 1) snpSubsetPath: String corresponding to the feature subset path
          2) validMafSnpsPath: String corresponding to cross-dataset consistent SNPs to narrow feature set to. 
                               If None, no further filtering is done to feature set.
          3) featherFilePath: String corresponding to the full dataset genotype feather file.

Output: genotypes: DataFrame of genotypes for a given dataset
'''
def LoadAndProcessGenotypes(snpSubsetPath: str, validMafSnpsPath: str, featherFilePath: str):
    # Get snp IDs from list
    snpIDs = list(set(pd.read_csv(snpSubsetPath, header = None, dtype = str)[0].tolist()))
    invalidSNPPath = "{}".format(os.path.join("/", *snpSubsetPath.split("/")[:-1], "InvalidSNPs.txt"))
    invalidSNPs = pd.read_csv(invalidSNPPath, header = None, dtype = str)[0].tolist()

    if len(invalidSNPs) >= 0:
        logger.info("{} SNPs exist across the full 5e-4 dataset with INF or NULL values. Removing these now... Unremoved SNP set size is {}".format(len(invalidSNPs), len(snpIDs)))
        snpIDs = [el for el in snpIDs if el not in invalidSNPs]
        logger.info("Cleaned SNP set size after removal of invalid SNPs is {}".format(len(snpIDs)))

    if validMafSnpsPath is not None:
        logger.info("Filtering SNP set for MAF and genotype consistent SNPs...")
        consistentMAFGenoSNPs = pd.read_csv(validMafSnpsPath, header = None, dtype = str)[0].tolist() 
        snpIDs = [el for el in snpIDs if el in consistentMAFGenoSNPs]
        logger.info("Cleaned SNP set size after filtering incompatible genotype and MAF discrepancy SNPs is {}".format(len(snpIDs)))

    # Load genotypes
    genotypes = feather.read_feather(featherFilePath).set_index('index')

    validSNPs = sorted(pd.Index(snpIDs)[pd.Index(snpIDs).isin(genotypes.index)])
    genotypes = genotypes.loc[validSNPs]
    genotypes = OrderSNPs(genotypes) #Order SNPs by CHR LOC 
    
    return genotypes 

'''
About: Add patch mapping to genotypes dataframe. Returns a genotypes DataFrame with
       a column "Patch", mapping a given SNP to its particular patch. .groupby("Patch")
       can be subsequently called on this for efficient mapping of patches to SNPs.
'''
def GeneratePatches(genotypes, patchMappingPath):
    patchMapping = joblib.load(patchMappingPath)
    reversePatchMapping = {snp:k for k,v in patchMapping.items() for snp in v}

    genotypes["Patch"] = genotypes.index.map(reversePatchMapping)

    splitPath = patchMappingPath.split("/")
    leftPatchPath = os.path.join("/".join(splitPath[:-1]), "_".join(splitPath[-1].split("_")[:2]) + "_Left_Shifted_Patches.joblib")
    rightPatchPath = os.path.join("/".join(splitPath[:-1]), "_".join(splitPath[-1].split("_")[:2]) + "_Right_Shifted_Patches.joblib")

    if os.path.exists(leftPatchPath) and os.path.exists(rightPatchPath):
        leftShiftPatchMapping = joblib.load(leftPatchPath)
        rightShiftPatchMapping = joblib.load(rightPatchPath)

        #Note - for all SNPs that do not have a left or right shift in genotypes will be mapped to a NaN which will be skipped by groupby
        genotypes["Left_Shifted_Patch"] = genotypes.index.map({snp:k for k,v in leftShiftPatchMapping.items() for snp in v})
        genotypes["Right_Shifted_Patch"] = genotypes.index.map({snp:k for k,v in rightShiftPatchMapping.items() for snp in v})

    return genotypes

'''
About: Method to processes a genotype DataFrame to individual-level dictionaries, with each dictionary
representing  particular datapoint/individuals genotype, with the keys as patches and the relevant
SNPs in that patch (the actual features) as the values. Processed dictionaries are then written to 
specified file path, which can be loaded directly during dataloading of a SNP_Dataset object.

Input(s): 1) genotypes: DataFrame corresponding to a dataset's loaded and preprocessed genotypes.
          2) patchMappingPath: String corresponding to where patch mappings are found.
          3) writePath: String corresponding to where to write individual-level files.
          4) datasetIDs: List of a dataset's IDs (for which to write files)
          5) sparsityThreshold: Integer corresponding to sparsity for which to omit certain patches.
          6) enableShifting: Boolean corresponding to whether shift patch tokenization desired. If so will update 
                             featurePatchSizes and featurePatchToSNPs with correct numbers and indices.
          7) cacheMethod: String in {'feather', 'joblib'} corresponding to genotype cache method.
          
Output: None
'''
def CachemAllPatchymon(genotypes, patchMappingPath, writePath, datasetIDs, sparsityThreshold, enableShifting, cacheMethod):
    genotypesWithPatches = GeneratePatches(genotypes = genotypes, patchMappingPath = patchMappingPath)
    fileNameFormat = "{datasetID}." + cacheMethod

    if enableShifting:
        fileNameFormat = "{datasetID}.joblib"
    
    #columnSubset = ["Patch"]
    #if "Left_Shifted_Patch" in genotypesWithPatches.columns:
    #    columnSubset = columnSubset + ["Left_Shifted_Patch", "Right_Shifted_Patch"]

    #<-- Replace this with a joblib dump
    
    #for point in datasetIDs: 
    #    feather.write_feather(df = genotypesWithPatches[[point] + columnSubset], dest = os.path.join(writePath, fileNameFormat.format(datasetID = point)))
        #joblib.dump({group: torch.from_numpy(data[point].values).float() for group, data in groupedGenotypesByPatches}, os.path.join(writePath, fileNameFormat.format(datasetID = point)))

    groupedGenotypesByPatches = genotypesWithPatches.groupby("Patch", observed = True)
    
    if enableShifting:
        # Instantiate unshifted genotypes patch sizes and patch -> SNPs mappings
        patchSizes = dict()
        featurePatchToSNPs = dict() #Added in case want access down the line
        for group, data in tqdm.tqdm(groupedGenotypesByPatches):
            numSNPsInPatch = data.shape[0]
            if numSNPsInPatch >= sparsityThreshold: #In case want to omit certain sparse patches (default is 0) so all patches of 1 or more SNPs will be included
                patchSizes[group] = numSNPsInPatch
                featurePatchToSNPs[group] = data.index.tolist() 

        # group left and right shifted patches
        leftShiftedGenos = genotypes.groupby("Left_Shifted_Patch", observed = True)
        rightShiftedGenos = genotypes.groupby("Right_Shifted_Patch", observed = True)

        #concatenate left shifted and right shifted patches with unshifted features
        patches = [patch for patch in featurePatchToSNPs.keys()]

        #NOTE: Doesn't account for whether a left or right shifted patch incorporates from a potentially skipped patch due to below sparsity threshold
        for patch in patches:
            if patch in leftShiftedGenos.groups.keys():
                featurePatchToSNPs[patch] = featurePatchToSNPs[patch] + leftShiftedGenos.get_group(patch).index.tolist()
            
            if patch in rightShiftedGenos.groups.keys():
                featurePatchToSNPs[patch] = featurePatchToSNPs[patch] + rightShiftedGenos.get_group(patch).index.tolist()

            patchSizes[patch] = len(featurePatchToSNPs[patch])

        
        joblib.dump(patchSizes, os.path.join(writePath, "featurePatchSizes.joblib")) 
        joblib.dump(featurePatchToSNPs, os.path.join(writePath, "featurePatchToSNPs.joblib")) 
        
    else:
        featurePatchToSNPs = {group:data.index.tolist() for group, data in groupedGenotypesByPatches if data.shape[0] >= sparsityThreshold}
        joblib.dump(featurePatchToSNPs, os.path.join(writePath, "featurePatchToSNPs.joblib"))
        joblib.dump({group:data.shape[0] for group, data in groupedGenotypesByPatches if data.shape[0] >= sparsityThreshold}, os.path.join(writePath, "featurePatchSizes.joblib"))
        
    
    if (not enableShifting) and (cacheMethod == "feather"):
        for point in datasetIDs: 
            feather.write_feather(df = genotypesWithPatches[[point, "Patch"]], dest = os.path.join(writePath, fileNameFormat.format(datasetID = point)))
    else: 
        #Parallelize writing of feature dictionaries:
        Parallel(n_jobs = -1)(delayed(_Parallelize_Feature__Dict_Write)(col_name, genotypes, featurePatchToSNPs, writePath, fileNameFormat) for col_name in datasetIDs)
    
    #for point in datasetIDs: 
    #    joblib.dump({patch: torch.from_numpy(genotypes.loc[snps, point].values).float() for patch, snps in featurePatchToSNPs.items()}, os.path.join(writePath, fileNameFormat.format(datasetID = point)))

    return None 

'''
About: Helper function for parallization of feature dict writing during caching
'''
def _Parallelize_Feature__Dict_Write(col_name, genotypes, feature_dict, write_path, file_name_format):
    features = {patch: torch.from_numpy(genotypes.loc[snps, col_name].values).float() for patch, snps in feature_dict.items()}
    
    joblib.dump(features, os.path.join(write_path, file_name_format.format(datasetID = col_name)))


'''
About: Converts a grouped by VADEr patch dataframe to a dictionary of IDs, patches, torch tensor of values.
Preprocessing during __init__ ensures don't have to initialize individual-level dictionary of features during 
non-cached loading.  

CURRENTLY OMITTED... Too slow and memory intensive to convert from grouped DF to individual representations
'''
def GroupedPatchesToDictionary(groupedDF, datasetIDs):
    genotypesAsDictionary = dict()

    patchSizes = {group:data.shape[0] for group, data in groupedDF}

    for point in tqdm.tqdm(datasetIDs):
        genotypesAsDictionary[point] = dict()
        for group, data in groupedDF:
            genotypesAsDictionary[point][group] = data[point].values #torch.from_numpy(data[point].values).float()
            
    return genotypesAsDictionary, patchSizes

'''
About: Method to validate correction of diagonal unmasking for single chromosome patches.
'''
def _Validate_LSA_Single_Chrom_Patch(singlePatchesPerChrom, mask, num_registers = 0):
    singlePatches = set(singlePatchesPerChrom)
    assert len(singlePatchesPerChrom) == len(singlePatches), "singlePatchesPerChrom has duplicate elements..."

    positionsToCorrect = torch.where(mask.logical_not().sum(dim = 1) == (2 + num_registers))[0]

    assert set([pos.item() for pos in positionsToCorrect]) == singlePatches, "INCONSISTENCY IN IDENTIFIED SINGLE CHROMOSOMAL PATCHES AND MASK CORRECTION SINGLE PATCHES. Exiting..."


'''
About: Method to generate a chromosome attention mask to pass to VADEr models. Specifically, this method generates a
mask that allows for attention only between patches on the same chromosome.

Input(s): 1) patch_to_chrom_mapping_path: String corresponding to the path   
          2) feature_patches: Dictionary whose keys correspond to feature patches in SNP_Dataset() object (can pass in the dataset patchSizes)
          3) num_registers: Integer corresponding to the number of registers employed in model.
Output(s): 1) chromMask: torch boolean tensor allowing for interactions between patches on the same chromosome (Falses correspond to entries that will not be masked in attention)
           2) singlePatchesPerChrom: List corresponding to the mask indices for which only a single patch exists per chromosome. These patches will cause problems with LPA which mask diagonal elements. If non-empty, diagonal mask (if using LPA) should be modified for these entries.
'''
def GenerateChromosomePatchMask(patch_to_chrom_mapping_path, feature_patches, num_registers = 0):
    if num_registers is None:
        num_registers = 0

    patchToChromMapping = pd.read_csv(patch_to_chrom_mapping_path, sep = "\t", index_col = 0)
    patchToChromMapping = patchToChromMapping[patchToChromMapping.index.isin(feature_patches.keys())]
    
    assert (patchToChromMapping == patchToChromMapping.sort_values(by = ["CHR", "ChromosomePatch"])).all().all(), "patchToChromMapping is not ordered according to [CHR, ChromosomePatch]. Reorder mapping and guarantee feature patches align with sorted mapping (i.e., clump# is ordered according to this ordering)."
    
    #Initialize chromosome mask:
    chromosomeMask = torch.ones(size = [len(feature_patches)]*2, dtype = torch.bool) #True everywhere
    
    #Get all positions where chromosome changes (e.g., position where CHR m+1 starts in transition from m to m+1)
    chrValues = patchToChromMapping["CHR"].to_numpy()
    chromChanges = np.where(chrValues[:-1] != chrValues[1:])[0] + 1
    singlePatchesPerChrom = list()
    
    # Update the mask to allow for interactions between patches only on the same chromosome 
    for i in range(len(chromChanges) + 1):
        if i == 0:
            assert chromChanges[i] > 0, "first position of chromChanges is 0 (must be > 0). Exiting..."
            start = 0
            end = chromChanges[i]
        elif i == len(chromChanges):
            start = chromChanges[i - 1] 
            end = patchToChromMapping.shape[0]
        else:
            start = chromChanges[i - 1]
            end = chromChanges[i]
        
        evalAllGood = patchToChromMapping.iloc[start:end, :].CHR
        assert len(set(evalAllGood)) == 1, "Issue in mask generation... Mask spans more than one chromosome."
        
        if end-start <= 1: 
            logging.warning(f"WARNING: CHR {set(evalAllGood)} has only one patch. Employing LPA will mask the entire chromosome representation.")
            singlePatchesPerChrom.append(start + 1)  #shift by 1 as chromMask will be adjusted to include for CLS at beginning
            
        chromosomeMask[start:end, start:end] = False
        
    #Add CLS column/row to beginning of mask as well as registers (all False)
    chromosomeMask = F.pad(chromosomeMask, (1, num_registers, 1, num_registers))
    
    #Validate LSA diagonal correction
    if len(singlePatchesPerChrom) > 0:
        _Validate_LSA_Single_Chrom_Patch(singlePatchesPerChrom, chromosomeMask, num_registers)
        

    return chromosomeMask