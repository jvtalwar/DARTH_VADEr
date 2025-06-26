'''
@author: James V. Talwar

About: DARTH_Computation.py is a script for generating DARTH (Directed Attention Relevance from Transformer Heuristics) scores for all checkpointed models in a 
trained VADEr model_checkpoint_path directory. This script takes in a --config_path argument that points to a .yaml file containing the configuration for trained 
VADEr model(s). The user does not need to provide a model_checkpoint_path argument, as this script will automatically find all model checkpoints in the provided
model_checkpoint_path directory specified in the config file specified at --config_path (as this was where model checkpoints were saved during training). 

python DARTH_Computation.py --config_path /path/to/config.yaml 
                            --test_feather_path /path/to/test.feather 
                            --test_phenos /path/to/test_phenos.tsv 
                            --test_ids /path/to/test_ids.txt
                            --write_dir /path/to/write_dir
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
import logging
import argparse

#import sys
#sys.path.append('./src/')

from VADErData import SNP_Dataset
from VADErDataUtils import GenerateChromosomePatchMask
from vader import VADEr

logger = logging.getLogger(__name__)

'''
About: Load configuration from yaml-config path

Input(s): path: String corresponding to path to yaml config file for training.
'''
def LoadConfig(path):
    return yaml.load(open(path, 'r'), Loader = yaml.SafeLoader)

def Initialize_Model(model_params, patch_sizes, training_objective, **kwargs):
    if training_objective == "sup_con":
        #Projection network dim initialized to model_params:disease as this should be the stage 2 trained VADEr model
        contrastive_projection_net_dim = model_params["prediction_dims"]["disease"]
    else:
        contrastive_projection_net_dim = None

    attention = model_params.get("attention")
    patchLayerNorm = model_params.get("patch_layer_norm")
    num_registers = model_params.get("num_registers")

    if attention is None:
        attention = "MHA"

    if patchLayerNorm is None:
        patchLayerNorm = attention == "LSA"

    if num_registers is None:
        num_registers = 0

    lord_vader = VADEr(patchSizes = patch_sizes,
                       modelDim = model_params["model_dim"],
                       mlpDim = model_params["model_dim"] * model_params["mlp_scale"],
                       depth = model_params["num_transformer_blocks"],
                       attnHeads = model_params["num_attention_heads"],
                       attnHeadDim = model_params["model_dim"]//model_params["num_attention_heads"],
                       multitaskOutputs = model_params["prediction_dims"],
                       clumpProjectionDropout = model_params["patch_projection_dropout"],
                       dropout = model_params["model_dropout"], 
                       ageInclusion = model_params["age_inclusion"],
                       aggr = model_params["aggregation"],
                       context = model_params.get("cls_representation"),
                       patchProjectionActivation = model_params["non_linear_patch_projection"],
                       patchLayerNorm = patchLayerNorm, 
                       trainingObjective = training_objective,
                       attention = attention,
                       numRegisters = num_registers,
                       ffActivation = model_params.get("mlp_method"), 
                       contrastive_projection_net_dim = contrastive_projection_net_dim)
    
    return lord_vader

def Average_Heads(attention, gradient, conjugate = False, **kwargs):
    attention_map = attention * gradient #[b, h, n, n]
    if conjugate:
        attention_map *= -1
    attention_map = attention_map.clamp(min = 0).mean(dim = 1) #[b, n, n]
    
    return attention_map

def Self_Attention_Rule(attention_map, relevance_map):
    return torch.matmul(attention_map, relevance_map)

def Generate_Relevance(model, batch_size, num_tokens, device, **kwargs):
    #Identify number of registers used in VADEr model
    try:
        num_registers = model.registers.size(1)
    except:
        num_registers = 0

    R = torch.eye(num_tokens, num_tokens).unsqueeze(0).repeat(batch_size, 1, 1).to(device) #[b, n, n]
    for block in model.transformer.blocks:
        attention = block[1].get_attention_map().detach()
        attention_gradient = block[1].get_attention_gradients()
        block[1].reset_attention_attributes()
        attention_map = Average_Heads(attention = attention, gradient = attention_gradient, **kwargs)
        R += Self_Attention_Rule(attention_map = attention_map, relevance_map = R)
    
    if num_registers > 0:
        return R[:, 0, 1:-num_registers].to("cpu")
    
    else:
        return R[:, 0, 1:].to("cpu")
    
def Generate_Transformer_Explainability(model, loader, device, mask = None, **kwargs):
    model.eval() 
    
    if mask is not None:
        mask = mask.to(device)

    vader_attribution = torch.Tensor()
    
    if "conjugate" in kwargs:
        assert kwargs["conjugate"] in {True, False}, "invalid option for conjugate. conjugate must be in {True, False}."
        logger.info(f"DARTH score conjugate status: {kwargs['conjugate']}")
        
    for i, (patchBatch, diseaseStatusBatch, ancestryBatch, fHBatch, zAgeBatch) in enumerate(loader):
        model.zero_grad()
        gpuClumpBatch = {k:v.to(device) for k,v in patchBatch.items()} #features
        
        if model.includeAge: #including age - need to pass in more than clump dictionary
            output = model(dictOfClumps = gpuClumpBatch, mask = mask, age_batch = zAgeBatch.to(device), extract_attention = True)
        else:
            output = model(dictOfClumps = gpuClumpBatch, mask = mask, extract_attention = True)
        
        #sum logits - dy/dA will then be computed for each element in the batch; logits used for cleaner gradients
        z = output["disease"].sum()
        
        #z = F.sigmoid(output["disease"]).sum() #<-- Sigmoid instead of logits if desired: scales derivative by g(x)*1-g(x)
        
        z.backward()
        
        batch_attribution = Generate_Relevance(model = model, 
                                               batch_size = output['disease'].shape[0], 
                                               num_tokens = model.transformer.blocks[0][1].get_attention_map().size(-1),
                                               device = device,
                                               **kwargs)
        
        vader_attribution = torch.cat([vader_attribution, batch_attribution])
        
        # Clean up to prevent memory buildup
        torch.cuda.empty_cache()
        
    return vader_attribution


def main(args):

    #Pull specs from config:
    config = LoadConfig(args.config_path)
    train_and_checkpoint = config["train_and_checkpoint"]
    model_params = config["model_params"]
    dataset_params = config["dataset"]

    if args.num_workers is None:
        num_workers = train_and_checkpoint["num_workers"]
    else:
        num_workers = args.num_workers
    
    if args.batch_size is None:
        batch_size = train_and_checkpoint["batch_size"]//2
    else:
        batch_size = args.batch_size

    #Initialize data/loader:
    test_dataset = SNP_Dataset(featherFilePath = args.test_feather_path,
                              phenoFilePath = args.test_phenos,
                              idFilePath = args.test_ids, 
                              snpSubsetPath = dataset_params["SNP_set"],
                              validMafSnpsPath = dataset_params.get("consistent_maf_SNPs"),
                              vaderPatchMappingPath = dataset_params["patch_mapping_path"],
                              trainingSetAgeStatsPath = dataset_params["age_train_stats"], 
                              sparsePatchThreshold = dataset_params.get("sparse_patch_threshold"),
                              enableShifting = train_and_checkpoint["shift_patches"])

    loader = DataLoader(dataset = test_dataset, pin_memory = True, shuffle = False, batch_size = batch_size, num_workers = num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #IDs and patch labels for DARTH scores:
    dataset_ids = pd.read_csv(args.test_ids, header = None, dtype = str)[0].tolist()
    numerical_ordered_patches = sorted([int(patch.split("p")[1]) for patch in test_dataset.patchSizes])
    patch_columns = ["patch" + str(el) for el in numerical_ordered_patches]

    #Define chromosome mask if was specified during training:
    chrom_mask = None
    if train_and_checkpoint.get("masking") == "chrom":
        assert dataset_params.get("patch_to_chrom_mapping") is not None, "Chromosome masking requires specification of patch_to_chrom_mapping in config dataset params."
        
        logger.info("Enabling chromosome masking")
        chrom_mask = GenerateChromosomePatchMask(patch_to_chrom_mapping_path = dataset_params.get("patch_to_chrom_mapping"),
                                                 feature_patches = test_dataset.patchSizes,
                                                 num_registers = model_params.get("num_registers"))

    #Get all checkpointed model full paths
    model_checkpoints = [os.path.join(train_and_checkpoint["model_checkpoint_path"], f_name) for f_name in os.listdir(train_and_checkpoint["model_checkpoint_path"]) if f_name.endswith('.pt')]
    logger.info(f"Identified {len(model_checkpoints)} VADEr model checkpoints in model_checkpoint_path: {train_and_checkpoint['model_checkpoint_path']}")
    
    if len(model_checkpoints) != train_and_checkpoint["k_checkpoint"]:
        logger.warning(f"Number of identified model checkpoints ({len(model_checkpoints)}) does not match CONFIG specified k_checkpoint value of {train_and_checkpoint['k_checkpoint']}")
    
    #Initialize model
    training_objective = "cross_entropy" if "cross_entropy" in train_and_checkpoint["loss"] else train_and_checkpoint["loss"]
    vader_model = Initialize_Model(model_params = model_params, 
                                       patch_sizes = test_dataset.patchSizes,
                                       training_objective = training_objective)
    
    logger.info(f"Beginning DARTH score computation for all models in {train_and_checkpoint['model_checkpoint_path']}...")

    for checkpoint_path in model_checkpoints:
        #Initialize VADEr and load weights:
        model_checkpoint_name = os.path.basename(checkpoint_path)
        logger.info(f"Loading VADEr checkpoint: {model_checkpoint_name}")
        
        vader_model.load_state_dict(torch.load(checkpoint_path)["modelStateDict"])
        vader_model.to(device)

        #if torch.cuda.device_count() > 1:
        #    torch.nn.DataParallel(vader_model, list(range(torch.cuda.device_count())))

        darth_vader_scores = Generate_Transformer_Explainability(model = vader_model, loader = loader, device = device, mask = chrom_mask, conjugate = args.conjugate)

        summary_scores = pd.DataFrame(darth_vader_scores, index = dataset_ids, columns = patch_columns)
        summary_scores.to_csv(os.path.join(args.write_dir, f"{model_checkpoint_name}_DARTH_scores.tsv"), sep = "\t")
        logger.info(f"DARTH scores computed for {model_checkpoint_name}...")

    return None 

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(message)s")

    parser = argparse.ArgumentParser()
    path_args = parser.add_argument_group("Input/output options:")
    path_args.add_argument("--config_path", type = str, required = True, help = "Path to VADEr config (.yaml file) for trained VADEr model(s).")
    path_args.add_argument("--test_feather_path", type = str, required = True, help = "Path to desired dataset genotype feather file for which want to compute DARTH scores.")
    path_args.add_argument("--test_phenos", type = str, required = True, help = "Path to dataset phenotypes for which want to compute DARTH scores.")
    path_args.add_argument("--test_ids", type = str, required = True, help = "Path to dataset IDs for which want to compute DARTH scores.")
    path_args.add_argument("--write_dir", type = str, required = True, help = "Directory to which write DARTH scores (one file per model checkpoint)")
    path_args.add_argument("--num_workers", type = int, default = None, help = "Number of workers for DataLoader. Default to config specified num_workers.")
    path_args.add_argument("--batch_size", type = int, default = None, help = "Batch size for evaluation. Default to config specified batch_size//2.")
    path_args.add_argument("--conjugate", action = "store_true", default = False, help = "Flag to specify generation of conjugate (disease protective) DARTH scores (i.e., negative relevance).")
    args = parser.parse_args()
    
    main(args)