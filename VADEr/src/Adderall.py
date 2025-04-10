'''
@author: James V. Talwar
Created on August 9, 2022 at 21:53:50/Reformatted on February 23, 2024

About: Adderall.py builds a multi-task Vision Adapted/Agnostic Disease Elucidating transforme-r (VADEr) for complex disease risk prediction. 
It is designed to take in genomic patches, with patches orderd by CHR-Base position (i.e., patch/clump 1 should equate to the lowest numbered
CHR in the dataset and all base pairs should fall in the patch radius as set during preprocessing) and return unactivated outputs for the 
multi-task prediction head. Task specific (or custom) losses, metrics, and activated predictions are assumed to be handled in 
training/fine-tuning/evaluation scripts that call the model. 
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
import math
from einops import repeat, rearrange, pack, unpack
import random
from TrainingUtils import PatchDropout, RMSNorm

logger = logging.getLogger(__name__)


class VADEr(nn.Module):
    '''
    INPUTS:
     patchSizes: dictionary mapping clumps to the number of SNPs in the clump
     modelDim: The genetic-patch projection size - this is the base dimension of the model (i.e., <CLS> and age representations will be this size as well) 
     mlpDim: The hidden layer dimension in each transformer MLP sub-block.
     attnHeads: The number of attention heads used in the multiheaded attention (Fluffy) blocks 
     attnHeadDim: The dimension of each of the attention heads in the Multiheaded attention (Fluffy) block
     depth: The number of transformer blocks (composed of attention and multi-layer perceptrons) in the model 
     multitaskOutputs: The number of output classes for the prediction head for each task (type dict) 
     clumpProjectionDropout: Dropout for clump projections (i.e. for each SNP clump to modelDim) + CLS + age. Applied before transformer. 
     dropout: Float corresponding to the dropout in the transformer blocks (in the attention and MLP blocks)
     ageInclusion: whether age is included in the architecure or not. This creates a age specific tensor of model dim and increases the model N from numClumps + 1 to numClumps + 2 
         (+ 1 comes from the inclusion of the CLS/context token); note when this is True, the forward method will need a key word argument of ageBatch = ... (default = False)
     aggr: String corresponding to whether want to mean/avg pool ("mean") or utilize the context vector (default - "cls")
     context: String in {"learnable", "global"}; "learnable" corresponds to a learnable CLS/context token while global corresponds to a CLS/context token derived from a linear
     projection from the full feature/SNP set (i.e., have all SNPs in the feature set compressed to model dim). Default is "learnable". Note None is also a viable option and if 
     passed in it is equivalent to the default "learnable".
     patchProjectionActivation: Boolean corresponding whether to apply non-linear (GELU) activation to SNP patch projections. Default False. 
     patchLayerNorm: Boolean corresponding to whether to apply layer norm before patch projections (default: False).
     trainingObjective: String in {cross_entropy, sup_con} corresponding to training objective. 'sup_con' will normalize embedding to be on the unit hypersphere before projection network
     numRegisters: Integer corresponding to the number of registers to use (for reasoning see VISION TRANSFORMERS NEED REGISTERS paper).
     ffActivation: String in {'GELU', 'SwiGLU'}, corresponding to MLP/FFN activation and implementation. Default: 'GELU'
     
    OUTPUT(S): A list of unactivated ouputs 
    '''
    
    def __init__(self, patchSizes: dict, modelDim: int, mlpDim: int, depth: int, attnHeads: int, attnHeadDim: int, multitaskOutputs: dict, 
                 clumpProjectionDropout: float = 0.0, dropout: float = 0.0, ageInclusion: bool = False, aggr: str = "cls", context = "learnable",
                 patchProjectionActivation: bool = False, patchLayerNorm = False, trainingObjective = "cross_entropy", attention = "MHA", 
                 numRegisters = 0, ffActivation = "GELU", **kwargs):
        super(VADEr, self).__init__() 
        
        assert context in {"learnable", "global", None}
        assert trainingObjective in {"cross_entropy", "sup_con"}, f"training objective {trainingObjective} invalid. Must be either 'cross_entropy' or 'sup_con'."
        assert attention in {"LSA", "MHA", "MHA-LT", "DA"}, f"Allowable attentions are 'MHA', 'MHA-LT' (MHA with learnable temperature), 'DA' (differential attention) and 'LSA'. Provided attention {attention} unsupported."
        assert ffActivation in {'GELU', 'SwiGLU', None}, f"ffActivation = {ffActivation} unsupported. Valid options are: 'GELU' and 'SwiGLU'"
        
        if ffActivation is None:
            ffActivation = "GELU"

        self.trainingObjective = trainingObjective

        self.clumpNum = sorted([int(k.split("p")[1]) for k in patchSizes]) #names are clump# (one-indexed)- get the numerical order of clumps. 
        
        self.patchProjection = nn.ModuleDict() # Project patchesclumps to modelDim.
        totalFeatureDim = 0 #total number of input features (i.e., SNPs here)
        for k,v in patchSizes.items(): 
            patchProj = [nn.Linear(v, modelDim)]
            
            if patchLayerNorm:
                patchProj = [nn.LayerNorm(v)] + patchProj

            if patchProjectionActivation:
                #Step 1: obtain latest predictions; Step 2: switch for GeLU vs. SwiGLU; Step 3: Implement SwiGLU
                patchProj.append(nn.GELU())
                
            if len(patchProj) == 1: #added for backwards model compatability 
                self.patchProjection[k] = patchProj[0] 
            else:
                self.patchProjection[k] = nn.Sequential(*patchProj)
                
            totalFeatureDim += v
            
        self.includeAge = ageInclusion
        
        if ageInclusion: #Wrap for memory efficiency (and DDP) - Won't be used in case of ageInclusion = False; 
            '''
            Note for 5e-8 ageless models this was not added so loaded state dicts will have this and may cause issues 
            though should be able to be handled at test inference time with something like: load_state_dict(state_dict, strict=False)
            '''
            logger.info("Including Age...")
            self.ageProjection = nn.Linear(1, modelDim)  
        
        if (context == "learnable") or (context is None):
            logger.info("Implementing LEARNABLE CLS token representation.")
            self.context = nn.Parameter(torch.randn(1, 1, modelDim))
        else:
            logger.info("Implementing PERSONALIZED CLS tokens (full SNP projection to model dim).")
            self.context = nn.Linear(totalFeatureDim, modelDim) #project all SNPs as a personalized CLS/context token representation
        
        if numRegisters > 0: #Initialize registers
            logger.info(f"Enabling {numRegisters} registers")
            self.registers = nn.Parameter(torch.randn(1, numRegisters, modelDim))
            self.registers_enabled = True

        else:
            self.registers_enabled = False 


        self.contextIsLearnable = isinstance(self.context, nn.Parameter)
        
        self.positionalEncoding = nn.Parameter(torch.randn(1, len(patchSizes) + 1 + int(ageInclusion), modelDim))
        self.clumpDropout = nn.Dropout(clumpProjectionDropout)
        
        self.transformer = Megatron(modelDim = modelDim, 
                                    blocks = depth, 
                                    mlpDim = mlpDim, 
                                    attnHeadDim = attnHeadDim, 
                                    numHeads = attnHeads, 
                                    dropout = dropout, 
                                    attention = attention,
                                    mlp_method = ffActivation,
                                    num_registers = numRegisters)
        self.aggr = aggr #how want to predict - on the context vector ("context") or mean pooling
        
        if trainingObjective == "sup_con":
            self.projectionNetwork = nn.Linear(modelDim, kwargs["contrastive_projection_net_dim"]) #nn.Sequential(nn.LayerNorm(modelDim), nn.Linear(modelDim, kwargs["contrastive_projection_net_dim"]))

        else:
            self.classificationHeads = nn.ModuleDict()
            for task, outputDimension in multitaskOutputs.items():
                taskHead = nn.Sequential(nn.LayerNorm(modelDim), nn.Linear(modelDim, outputDimension))
                self.classificationHeads[task] = taskHead
        
    def forward(self, dictOfClumps, mask = None, patch_dropout = False, extract_attention = False, **kwargs):  #when age is included a key word of ageBatch = ... is needed
        x = torch.Tensor().to(self.positionalEncoding.device)
        allSNPs = torch.Tensor().to(self.positionalEncoding.device)
        
        for clumpNumber in (self.clumpNum): 
            key = "clump" + str(clumpNumber)
            x = torch.cat([x, self.patchProjection[key](dictOfClumps[key]).unsqueeze(1)], dim = 1)
            if not self.contextIsLearnable:
                allSNPs = torch.cat([allSNPs, dictOfClumps[key]], dim = 1)
                
        if self.includeAge:
            x = torch.cat([self.ageProjection(kwargs["age_batch"]).unsqueeze(1), x], dim = 1) #another option might be to add the age projection to the context vector rather than a independent vector/tensor
            
        batchSize, n, model_dim = x.shape #n = number of genetic patches + int(ageIncluded)  
        
        if self.contextIsLearnable:
            contextTokens = repeat(self.context, '1 1 d -> b 1 d', b = batchSize)
        else:
            contextTokens = self.context(allSNPs).unsqueeze(1)
            del allSNPs
            
        x = torch.cat([contextTokens, x], dim = 1) 
        
        #incorporate the positional information and clump projection dropout
        x += self.positionalEncoding

        
        if patch_dropout:
            fractionTokensToKeep = random.random()

            # Half the time dropout a random subset of patches from 0.5 - 1; otherwise use all patches
            if fractionTokensToKeep < 0.5:
                fractionTokensToKeep = 1 
            
            x, patch_mask = PatchDropout(x = x, inference = not self.training, keep_ratio = fractionTokensToKeep)

            #If mask employed, need to ensure correctly update mask so dimensions are correct
            if (mask is not None) and (patch_mask is not None):
                if self.registers_enabled: #add back registers if enabled into the mask
                    #n+1 patches including CLS - indexing for first register then is n + 1
                    registerPositions = torch.arange(n + 1, n + 1 + self.registers.shape[1]).unsqueeze(0).unsqueeze(2).expand(batchSize, -1, model_dim)

                    #Concatenate registerPositions to patch_mask along patch dimension
                    patch_mask = torch.cat([patch_mask, registerPositions], dim = 1)
                                    
                mask = torch.gather(mask, dim = 1, index = patch_mask)
                
        
        x = self.clumpDropout(x) #an important note: dropout scales outputs by a factor of 1/(1-p) during training which if unaware can make tracebacks infuriating!
        
        #Add registers as desired:
        if self.registers_enabled:
            registers = repeat(self.registers, "1 k d -> b k d", b = batchSize) #k -> numRegisters
            x, ps = pack([x, registers], "b * d") #ps == [numPatches + 1, numRegisters]

        x = self.transformer(x, mask, extract_attention)

        #Remove registers:
        if self.registers_enabled:
            x, _ = unpack(x, ps, "b * d")

        x = x.mean(dim = 1) if self.aggr == "mean" else x[:, 0] #[B, D]
        
        
        if self.trainingObjective == "sup_con":
            #normalize x to fall on unit hypersphere
            x = F.normalize(x, dim = -1, p = 2) 

            #Get embedding by projection network (un-normalized); normalization of projection network features will be handled by supervised contrastive loss
            x = self.projectionNetwork(x)

            return x

        else:
            mtlOutputs = dict()
            for task, classificationLayer in self.classificationHeads.items():
                mtlOutputs[task] = classificationLayer(x)
            
            #print("\tIn Model: input size", x.size(),"output size", mtlOutputs[0].size())
            
            return mtlOutputs
    

#MLP (formely - Sherlock): MLP in a transformer block. Why Sherlock? Well if you have to ask than you may see but do not perceive/observe!
class MLP(nn.Sequential): 
    '''Subclassing Sequential --> don't need to write a forward f(x)'''
    
    def __init__(self, modelDim: int, mlpDim: int, dropout: float = 0.0): 
        super().__init__(nn.Linear(modelDim, mlpDim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlpDim, modelDim), nn.Dropout(dropout))
        

class SwiGLU(nn.Module):
    def __init__(self, modelDim: int, mlpDim: int, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(modelDim, mlpDim * 2)
        self.outputProjection = nn.Linear(mlpDim, modelDim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, gate = self.gate(x).chunk(2, dim = -1)
        x = x * F.silu(gate)
        #x = self.dropout(x)
        x = self.outputProjection(x)
        x = self.dropout(x)

        return x

#MHA (formerly Fluffy): Multi-headed attention in a transformer block. Why Fluffy? Because a many-headed dog is paying attention! "Ain't nobody gettin past fluffy..." (unless you play him a bit of music)
class MHA(nn.Module):
    def __init__(self, modelDim: int, headDim: int, dropout: float = 0.0, numHeads: int = 8, learnableTemp = False):
        super().__init__()
        
        projectionDim = numHeads * headDim #the dimension we will project our modelDim inputs to for attention
        needFinalProjection = not ((projectionDim == modelDim) and (numHeads == 1)) #in this case the dimensions are compatible with modelDim
        
        self.heads = numHeads
        self.learnableTemp = learnableTemp
        if learnableTemp:
            self.scaleFactor = nn.Parameter(torch.log(torch.tensor(headDim ** -0.5)))
        else:
            self.scaleFactor = headDim ** -0.5
        
        self.qkv = nn.Linear(modelDim, projectionDim * 3, bias = False)
        
        self.softmax = nn.Softmax(dim = -1)
        self.withTheDropout = nn.Dropout(dropout)
        
        self.projectionLayer = nn.Sequential(nn.Linear(projectionDim, modelDim), nn.Dropout(dropout)) if needFinalProjection else nn.Identity()
        
        self.attentionMap = None
        self.attentionGradients = None
        self.gradients_hook_handle = None

    def save_attention_gradients(self, attention_gradients):
        self.attentionGradients = attention_gradients.detach()

    def get_attention_map(self):
        return self.attentionMap
    
    def get_attention_gradients(self):
        return self.attentionGradients

    def reset_attention_attributes(self):
        self.attentionMap = None
        self.attentionGradients = None
        if self.gradients_hook_handle is not None:
             self.gradients_hook_handle.remove()
             self.gradients_hook_handle = None
    
    def forward(self, x, mask = None, extract_attention = False):
        qkv = self.qkv(x).chunk(3, dim = -1)  #qkv is a tuple of tensors - need to map to extract individual q,k,v
        
        '''
        Unpack the map object to the requisite tensors; map applies a function on the individual elements of the structure here a tuple; 
        einops.rearrange() makes dimension manipulation easy for converting to multi-headed attention
        '''
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  
        
        if self.learnableTemp:
            scaler = self.scaleFactor.exp()
        else:
            scaler = self.scaleFactor

        scaledScore = torch.matmul(q, k.transpose(-1, -2)) * scaler

        #Allow for (chromosomal patch) masking:
        if mask is not None:
            scaledScore = scaledScore.masked_fill(mask, torch.finfo(scaledScore.dtype).min) #mask.to(scaledScore.device)

        attention = self.softmax(scaledScore) #attention sums to 1 along the last dimension; Ensures weighted average accumulation across the values (i.e., attention)
        attention = self.withTheDropout(attention)
        
        if extract_attention:
            self.attentionMap = attention
            self.gradients_hook_handle = attention.register_hook(self.save_attention_gradients)
    

        output = torch.matmul(attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projectionLayer(output)
        return output
    
#Locality Self-Attention
class LSA(nn.Module):
    def __init__(self, modelDim: int, headDim: int, dropout: float = 0.0, numHeads: int = 8, num_registers: int = 0):
        super().__init__()
        
        self.num_registers = num_registers #needed  for correction of LSA 
        projectionDim = numHeads * headDim #the dimension we will project our modelDim inputs to for attention
        needFinalProjection = not ((projectionDim == modelDim) and (numHeads == 1)) #in this case the dimensions are compatible with modelDim
        
        self.heads = numHeads
        self.scaleFactor = nn.Parameter(torch.log(torch.tensor(headDim ** -0.5))) #Learnable temperature in LSA
        self.qkv = nn.Linear(modelDim, projectionDim * 3, bias = False)
        
        self.softmax = nn.Softmax(dim = -1)
        self.withTheDropout = nn.Dropout(dropout)
        
        self.projectionLayer = nn.Sequential(nn.Linear(projectionDim, modelDim), nn.Dropout(dropout)) if needFinalProjection else nn.Identity()
        
        self.attentionMap = None
        self.attentionGradients = None
        self.gradients_hook_handle = None

    def save_attention_gradients(self, attention_gradients):
        self.attentionGradients = attention_gradients.detach()

    def get_attention_map(self):
        return self.attentionMap
    
    def get_attention_gradients(self):
        return self.attentionGradients 
    
    def reset_attention_attributes(self):
        self.attentionMap = None
        self.attentionGradients = None
        if self.gradients_hook_handle is not None:
             self.gradients_hook_handle.remove()
             self.gradients_hook_handle = None

    def forward(self, x, mask = None, extract_attention = False): 
        qkv = self.qkv(x).chunk(3, dim = -1)  #qkv is a tuple of tensors - need to map to extract individual q,k,v
        
        '''
        Unpack the map object to the requisite tensors; map applies a function on the individual elements of the structure here a tuple; 
        einops.rearrange() makes dimension manipulation easy for converting to multi-headed attention
        '''
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)  
        
        scaledScore = torch.matmul(q, k.transpose(-1, -2)) * self.scaleFactor.exp()
        
        #Allow for (chromosomal patch) masking:
        positionsToCorrect = torch.tensor([])
        if mask is not None:
            scaledScore = scaledScore.masked_fill(mask, torch.finfo(scaledScore.dtype).min)
            
            #Identify any single patches - if exist, need to ensure reconstruction of these patches from self are allowed! 
            # == 2 + number of registers for Falses for CLS, self, and, number of registers.
            positionsToCorrect = torch.where(mask.logical_not().sum(dim = 1) == (2 + self.num_registers))[0] 

        #Diagonal LSA masking:
        diagonalMask = torch.eye(scaledScore.shape[-1], device = scaledScore.device, dtype = torch.bool)
        if positionsToCorrect.shape[0] > 0:
            diagonalMask[positionsToCorrect, positionsToCorrect] = False

        scaledScore = scaledScore.masked_fill(diagonalMask, torch.finfo(scaledScore.dtype).min)

        attention = self.softmax(scaledScore) #attention sums to 1 along the last dimension; Ensures weighted average accumulation across the values (i.e., attention)
        attention = self.withTheDropout(attention)
        
        if extract_attention:
            self.attentionMap = attention
            attention.register_hook(self.save_attention_gradients)

        output = torch.matmul(attention, v)
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projectionLayer(output)

        return output
    
def Differential_Attn_Lambda_Init_Fx(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class Differential_Attention(nn.Module):
    '''
    headDim = Integer corresponding to the dimension of V and thus in the paper convention equates to 2d
    '''
    def __init__(self, modelDim: int, headDim: int, transformer_layer: int, dropout: float = 0.0, numHeads: int = 8):
        super().__init__()
        
        #assert headDim % 2 == 0, f"headDim {headDim} corresponding to 2d not divisible by 2 which causes issues for Q,K splitting. Exiting..."
        
        projectionDim = numHeads * headDim #the dimension we will project our modelDim inputs to for attention
        
        self.heads = numHeads
        self.head_dim_qk = headDim//2 #q1, q2, k1, k2 dimension 
        self.scaleFactor = self.head_dim_qk**-0.5 #Differential attention - Q and K are partitioned into dimension d == projectionDim/2 
        
        self.qkv = nn.Linear(modelDim, projectionDim * 3, bias = False) 
        self.projectionLayer = nn.Sequential(nn.Linear(projectionDim, modelDim, bias = False), nn.Dropout(dropout)) 
        self.head_layer_norm = RMSNorm(headDim)
        
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = -1)
        
        #Define lambdas for differential attention:
        self.lambda_init = Differential_Attn_Lambda_Init_Fx(transformer_layer)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim_qk, dtype=torch.float32).normal_(mean=0,std=0.1))
        
    def forward(self, x, mask = None, extract_attention = False):
        qkv = self.qkv(x).chunk(3, dim = -1)  #qkv is a tuple of tensors - need to map to extract individual q,k,v
        
        '''
        Unpack the map object to the requisite tensors; map applies a function on the individual elements of the structure here a tuple; 
        einops.rearrange() makes dimension manipulation easy for converting to multi-headed attention
        '''
        q_merged, k_merged, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv) 
        
        #Generate q1,q2,k1,k2 - dimension d//2
        q1, q2 = q_merged.chunk(2, dim = -1)
        k1, k2 = k_merged.chunk(2, dim = -1)
        
        #Generate pre-softmaxed attentions
        scaledScore = torch.matmul(q1, k1.transpose(-1, -2)) * self.scaleFactor
        differentialScaledScore = torch.matmul(q2, k2.transpose(-1, -2)) * self.scaleFactor

        #Allow for (chromosomal patch) masking:
        if mask is not None:
            scaledScore = scaledScore.masked_fill(mask, torch.finfo(scaledScore.dtype).min) 
            differentialScaledScore = differentialScaledScore.masked_fill(mask, torch.finfo(scaledScore.dtype).min) 
            
        #Softmax attentions:
        mainAttention = self.softmax(scaledScore) #attention sums to 1 along the last dimension; Ensures weighted average accumulation across the values (i.e., attention)
        differentialAttention = self.softmax(differentialScaledScore)
        
        #Update lambda for subtraction:
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()) #.type_as(q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()) #.type_as(q2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attention = mainAttention - lambda_full * differentialAttention
        attention = self.dropout(attention)
        
        output = torch.matmul(attention, v)
        
        #Headwise normalization:
        output = self.head_layer_norm(output)
        output = output * (1 - self.lambda_init)
        
        output = rearrange(output, 'b h n d -> b n (h d)')
        output = self.projectionLayer(output)
        
        return output

#Megatron: The lead decepticon (or the Transformer object). Transforming to constantly cause problems for the Witwickys
class Megatron(nn.Module):
    def __init__(self, modelDim: int, blocks: int, mlpDim: int, attnHeadDim: int, numHeads: int = 8, dropout: float = 0.0, attention: str = "MHA", mlp_method: str = "GELU", num_registers: int = 0):
        super().__init__()

        if mlp_method == "SwiGLU":
            #Correct MLP-dim to account for the extra weight matrix in SwiGLU
            mlpDim = int(mlpDim * 2/3)
            logger.info(f"Implementing SwiGLU... correcting mlpDim to {mlpDim} to keep number params consistent")

        self.blocks = nn.ModuleList([])
        for i in range(blocks):
            if attention == "LSA":
                if mlp_method == "SwiGLU":                    
                    blockComponents = nn.ModuleList([nn.LayerNorm(modelDim), LSA(modelDim = modelDim, headDim = attnHeadDim, numHeads = numHeads, dropout = dropout, num_registers = num_registers), nn.LayerNorm(modelDim), SwiGLU(modelDim = modelDim, mlpDim = mlpDim, dropout = dropout)])

                else:
                    blockComponents = nn.ModuleList([nn.LayerNorm(modelDim), LSA(modelDim = modelDim, headDim = attnHeadDim, numHeads = numHeads, dropout = dropout, num_registers = num_registers), nn.LayerNorm(modelDim), MLP(modelDim = modelDim, mlpDim = mlpDim, dropout = dropout)])
            
            elif attention == "DA":
                logger.info("Differential Attention Enabled...")
                if mlp_method == "SwiGLU":                    
                    blockComponents = nn.ModuleList([RMSNorm(modelDim), Differential_Attention(modelDim = modelDim, headDim = attnHeadDim, numHeads = numHeads, dropout = dropout, transformer_layer = i+1), RMSNorm(modelDim), SwiGLU(modelDim = modelDim, mlpDim = mlpDim, dropout = dropout)])

                else:
                    blockComponents = nn.ModuleList([RMSNorm(modelDim), Differential_Attention(modelDim = modelDim, headDim = attnHeadDim, numHeads = numHeads, dropout = dropout, transformer_layer = i+1), RMSNorm(modelDim), MLP(modelDim = modelDim, mlpDim = mlpDim, dropout = dropout)])
            
            else:
                learnableTemperature = attention[-2:] == "LT"
                logger.info(f"Implementing learnable temperature for MHA: {learnableTemperature}")
                if mlp_method == "SwiGLU":
                    blockComponents = nn.ModuleList([nn.LayerNorm(modelDim), MHA(modelDim = modelDim, headDim = attnHeadDim, numHeads = numHeads, dropout = dropout, learnableTemp = learnableTemperature), nn.LayerNorm(modelDim), SwiGLU(modelDim = modelDim, mlpDim = mlpDim, dropout = dropout)])
                else:
                    blockComponents = nn.ModuleList([nn.LayerNorm(modelDim), MHA(modelDim = modelDim, headDim = attnHeadDim, numHeads = numHeads, dropout = dropout, learnableTemp = learnableTemperature), nn.LayerNorm(modelDim), MLP(modelDim = modelDim, mlpDim = mlpDim, dropout = dropout)])
            
            self.blocks.append(blockComponents)
            
    def forward(self, x, mask = None, extract_attention = False):
        for l1, attention, l2, mlp in self.blocks:
            x = attention(l1(x), mask, extract_attention) + x #layer norm first (i.e. pre-norm) then attention then add residual
            x = mlp(l2(x)) + x #layer norm (i.e. pre-norm) then mlp then add residual
        
        return x