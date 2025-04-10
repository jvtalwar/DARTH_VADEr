'''
@author: James V. Talwar
Created on 4/11/2024 at 19:27:31

About: TrainingUtils.py contains accessory classes and functions that may be useful when training VADEr models (e.g., SupConv, MixUp).
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot as OneHot
import random

'''
INPUT(s):
 1) hardTargets: Torch tensor of label targets to be smoother. Formatted either as 0/1 (binary), 
    one-hot (e.g., [0, 0, 1]; categorical), or one-hot indexes (e.g. 1 for [0,1,0]; categoricalIndexes)
 2) alpha: Float corresponding to smoothing parameter
 3) encoding: String in {'binary', 'categorical', 'categoricalIndexes'} corresponding to the target encoding
 4) **kwargs: key word arguments - numClasses (int) is required for non-binary encodings
 
OUTPUT(s):
 1) A tensor of smoothed labels
 
A method to smooth labels to minimize model overconfidence and improve generalization. 
FORMULA: ySmoothed = (1-alpha)*y + alpha/numClasses (https://arxiv.org/pdf/1906.02629.pdf)

ex: binary labels [0,1] with alpha == 0.2 --> [0.1, 0.9]
'''
def SmoothLabels(hardTargets, alpha = 0.1, encoding = "binary", **kwargs):
    if encoding not in {'binary', 'categorical', 'categoricalIndexes'}:
        raise ValueError("Invalid Encoding: Valid options are {'binary', 'categorical', 'categoricalIndexes'}.")
        
    if encoding == 'categoricalIndexes':
        hardTargets = OneHot(hardTargets.long(), num_classes = kwargs["numClasses"])
        ySmoothed = (1 - alpha) * hardTargets + alpha/kwargs["numClasses"]
        
    if encoding == "binary":
        ySmoothed = (1 - alpha) * hardTargets + alpha/2
            
    else:
        assert kwargs["numClasses"] == hardTargets.shape[-1] 
        
        ySmoothed = (1 - alpha) * hardTargets + alpha/kwargs["numClasses"] #
        
    return ySmoothed 



'''
About: Helper function to generate shuffled features and labels for VADEr, features and disease labels.
Called by MixUp() method.
'''
def _Shuffle(features, labels, ancestry = None, age = None):
    shuffledIndices = torch.randperm(labels.size(0), device = labels.device, dtype = torch.long)
    
    shuffledLabels = labels[shuffledIndices]
    shuffledFeatures = {k:v[shuffledIndices] for k,v in features.items()}

    if ancestry is None:
        shuffledAncestry = None
    else:
        shuffledAncestry = ancestry[shuffledIndices]

    if age is None:
        shuffledAge = None
    else:
        shuffledAge = age[shuffledIndices]
    
    return shuffledFeatures, shuffledLabels, shuffledAncestry, shuffledAge

'''
About: Implements MixUp as defined by mixup paper -->: mixup: BEYOND EMPIRICAL RISK MINIMIZATION

Input(s): 1) features: A dictionary of VADEr features mapping patches to underlying genotype feature tensors
          2) labels: A tensor of disease (i.e., 0/1) labels --> [B, 1]
          3) alpha: Integer corresponding to the alpha/beta for beta distribution.
          4) ancestry: Tensor of index (e.g., [0, 1, 0, 2, ...] - [bx1]) ancestry labels. Default None.
          5) num_ancestry_classes: An integer corresponding to the number of ancestry classes. Needed when ancestry is not None, to 
                                  convert from index representation to one-hot representation. Default: 5 
                                  (number of ancestries in PC dataset; BC has 3 ancestries). 
          6) age: Tensor of (z-scored) ages. Default: None.

Outputs: 1) mixedUpFeatures: A dictionary of mixed up features 
         2) mixedUpLabels: A tensor of mixed up labels
         
'''
def MixUp(features, labels, alpha = 1, ancestry = None, num_ancestry_classes = 5, age = None):
    lamduh = np.random.beta(alpha, alpha)
    
    shuffledFeatures, shuffledLabels, shuffledAncestry, shuffledAge = _Shuffle(features, labels, ancestry, age)
        
    mixedUpLabels = labels.mul(lamduh).add(shuffledLabels, alpha = 1 - lamduh)
    mixedUpFeatures = {k:v.mul(lamduh).add(shuffledFeatures[k], alpha = 1 - lamduh) for k,v in features.items()}
    
    if shuffledAncestry is not None:
        assert isinstance(num_ancestry_classes, int), "num_ancestry_classes for MixUp must be an integer."

        #Convert ancestry and shuffledAncestry to one hot vectors:
        ohAncestry = OneHot(ancestry.squeeze(1).long(), num_classes = num_ancestry_classes)
        ohShuffledAncestry = OneHot(shuffledAncestry.squeeze(1).long(), num_classes = num_ancestry_classes)

        mixedUpAncestry = ohAncestry.mul(lamduh).add(ohShuffledAncestry, alpha = 1 - lamduh)

    else:
        mixedUpAncestry = shuffledAncestry

    if shuffledAge is not None:
        mixedUpAge = age.mul(lamduh).add(shuffledAge, alpha = 1 - lamduh)
    else:
        mixedUpAge = shuffledAge


    return mixedUpFeatures, mixedUpLabels, mixedUpAncestry, mixedUpAge

'''
About: Implements MixAndMatch - a cross between PatchDropout and CutMix. Specifically, shift labels according to
       selected Beta distribution (like mixup), but rather than linearly combine features, replace a number patches 
       equating to the complement (i.e. 1-lambda) of the selected beta distribution value. 

Input(s):  1) features: A dictionary of VADEr features mapping patches to underlying genotype feature tensors
           2) labels: A tensor of disease (i.e., 0/1) labels --> [B, 1]
           3) alpha: A float value to parameterize the beta distribution with for sampling
           4) ensure_swapping: A Boolean value corresponding as to whether at least one patch should be swapped 
              (in cases where int(complement * numPatches) == 0). Default False 
Output(s): 1)  features: A dictionary of mixed and matched (by patches) features 
           2)  mixedUpLabels: A tensor of mixed up labels
'''
def MixAndMatch(features, labels, alpha = 1, ensure_swapping = False):
    lamduh = np.random.beta(alpha, alpha)
    
    shuffledFeatures, shuffledLabels, _ = _Shuffle(features, labels)
    mixedUpLabels = labels.mul(lamduh).add(shuffledLabels, alpha = 1 - lamduh)
    
    
    complement = 1 - lamduh #replaced patches should amount to (1-lamda) total fraction of patches
    patches = [k for k in features.keys()]
    numPatchesToSwap = int(len(patches) * complement)
    if ensure_swapping:
        numPatchesToSwap = max(1, numPatchesToSwap)
    
    #swap patches
    patchesToSwap = random.sample(patches, numPatchesToSwap)
    for patch in patchesToSwap:
        features[patch] = shuffledFeatures[patch]

    return features, mixedUpLabels


'''
About: Method to implement patch dropout, which omits a certain number of patches post-positional incorporation before
       the forward pass through transformer. 

Input(s): 1) x: torch tensor of CLS Token + patches - dims: b, n+1, d
          2) inference: Boolean corresponding whether are in inference (i.e., model.eval()) or training. 
             If True, no patches will be omitted/dropped-out
          3) keep_ratio: float corresponding to the fraction of patches to keep during inference. If 1, all patches are kept
Output(s): 1) x: torch tensor of CLS Token + randomly sampled patches - dims: b, int((n)*keep_ratio)) + 1, d
           2) patch_mask: torch tensor of positions kept - dims match output dim of x 
'''
def PatchDropout(x, inference, keep_ratio):
    if (inference) or (keep_ratio == 1):
        #print(f"INFERENCE {inference}; Keep Ratio {keep_ratio}")
        return x, None
    
    bs, numPatchesPlusCls, d, = x.shape
    keep = int((numPatchesPlusCls - 1) * keep_ratio) #number of patches to keep

    # Generate a random [b x n] tensor sort by index values and keep all those (in order) up to keep 
    patch_mask = torch.rand(bs, numPatchesPlusCls - 1, device = x.device) 
    patch_mask = torch.argsort(patch_mask, dim = 1) + 1 #shift by 1 as CLS is pre-pended
    patch_mask = patch_mask[:, :keep]
    patch_mask = patch_mask.sort(1)[0]

    cls_mask = torch.zeros(bs, 1, dtype = torch.int64, device = x.device)
    
    # cat cls and patch mask so cls tokens are always included in sub-sampling
    patch_mask = torch.hstack([cls_mask, patch_mask]) 

    #Expand along model dimension d
    patch_mask = patch_mask.unsqueeze(-1).repeat(1, 1, d)
    
    #Sub-select from x along the indices specified by patch_mask
    x = torch.gather(x, dim = 1, index = patch_mask)

    return x, patch_mask

'''
About: Method to gather VADEr (projection network) features across GPUs to a single tensor. 
'''
def GatherFeatures(featureTensor):
    #Output feature tensor should be (B * WorldSize) x d
    allFeatures = torch.zeros(featureTensor.shape[0] * torch.distributed.get_world_size(), 
                              featureTensor.shape[1], dtype = featureTensor.dtype, 
                              device = featureTensor.device)
    
    torch.distributed.all_gather_into_tensor(output_tensor= allFeatures, input_tensor = featureTensor)
    
    return allFeatures 

@torch.no_grad()
def GatherLabels(labelTensor):
    gatherLabelList = [torch.ones_like(labelTensor) for i in range(torch.distributed.get_world_size())]

    torch.distributed.all_gather(tensor_list = gatherLabelList,
                                 tensor = labelTensor)
    
    allLabels = torch.cat(gatherLabelList, dim = 0).to(labelTensor.device)

    return allLabels

'''
About: Method to subtract the maximum value along the last dimension of a tensor. Called for numerical stability
before calling log_softmax ()
'''
def StabilizeLogits(logits):
    maxLogits, _ = torch.max(logits, dim = -1, keepdim = True)
    logits = logits - maxLogits.detach()
    
    return logits

'''
About: Helper method for computing supervised contrastive loss. Computes the log softmax of the self-oblated logits
       (which handles the interior term of loss), then sums the relevant positives (i.e. instances of the same label)
       and normalizes by the number of positives. Finally loss is multiplied by -1 and normalization by the number of 
       samples (in my local batch).

INPUTS: logits: Tensor (dtype = float) of logits (dot product of features and allFeatures/temperature). Diagonal values (i.e. self) are filled
                with min value/large negative number to mask contribution. Dimensions: B x (B*WorldSize)
        positives: Tensor (dtype = float) of all labels that match a given point's label. Normalized by the number of positive instances. 
                   Dimensions: B x (B*WorldSize)

OUTPUT: loss: Tensor (dtype = Float) corresponding to supervised contrastive loss value. Dimensions: 1
'''
def CalculateCrossEntropy(logits, positives):
    interior = F.log_softmax(logits, dim = -1)
    summedAndNormalized = torch.sum(positives * interior, dim = -1) # Dim: [B]; Element-wise multiplication
    loss = - summedAndNormalized.mean() 

    return loss

class SupCon(nn.Module):
    '''
    Supervised contrastive loss as reported by: https://arxiv.org/pdf/2004.11362.pdf 

    Helpful repositories: 1) https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py#L49
                          2) https://github.com/HobbitLong/SupContrast/blob/master/losses.py 

    INPUTS:
     device_rank: Integer corresponding to GPU (global) rank.
     temperature: Float corresponding to the temperature to use in the supervised contrastive loss function
    '''
    def __init__(self, device_rank, temperature = 0.07): #0.1
        super(SupCon, self).__init__() 
        self.device_rank = device_rank
        self.temperature = temperature
    
    def forward(self, features, labels):
        # features: [B x d] 
        # labels: [B x 1]
        
        batchSize = features.shape[0]
        features = F.normalize(features, dim = -1, p = 2) #Output of projection network is normalized to lie on the unit hypersphere

        #Gather all features and labels onto current device
        allFeatures = GatherFeatures(features) #(B * WorldSize) x d
        allLabels = GatherLabels(labels) # (B * WorldSize) x 1

        #print(f"Device {self.device_rank} labels shape {labels.shape}")
        #print(f"Device {self.device_rank} allLabels shape {allLabels.shape}")

        #Define a mask for labels of the same class: B x (B * WorldSize)
        mask = torch.eq(labels.view(-1, 1), allLabels.contiguous().view(1, -1)).float().to(features.device) #for every row in labels (on current device), check against all labels to see if same

        #print(f"Device {self.device_rank} mask: {mask}")

        #print(f"Device {self.device_rank} Mask shape {mask.shape}")

        #Define a mask for anchors - for all N points, only N-1 points should be included (i.e., omit self in computation)
        anchorMask = torch.scatter(torch.ones_like(mask),  1, torch.arange(mask.shape[0]).view(-1, 1).to(features.device) + batchSize * self.device_rank,  0)

        #print(f"Device {self.device_rank} anchor mask: {anchorMask}")

        #Complete mask are those values either not of the same label or self
        mask = mask * anchorMask

        #print(f"Device {self.device_rank} combined mask: {mask}")
        #print(f"Device {self.device_rank} anchor mask shape {anchorMask.shape}")

        #Compute logits:
        logits = torch.matmul(features, allFeatures.T) / self.temperature
        
        #print(f"Device {self.device_rank} logits shape {logits.shape}")

        #If observe inf or nans can replace torch.finfo(logits.dtype).max with 1e9 (or something of the like)
        logits = logits - (1 - anchorMask) * torch.finfo(logits.dtype).max #Remove self logits (i.e., logits of a given point in a batch)

        #print(f"Device {self.device_rank} logits pre-stabilized: {logits}")

        # For numerical stability (subtract max along dim = -1; LSE trick); Potentially optional... (see helpful repo 1 above)
        # Note: Max possible value given feature normalization is 1/self.temperature
        logits = StabilizeLogits(logits)

        #print(f"Device {self.device_rank} logits post-stabilized: {logits}")

        #normalize mask by the number of labels matching self (and clamping denom min to 1 to prevent div by 0)
        mask = mask/mask.sum(dim = -1, keepdim = True).clamp(min = 1)

        #print(f"Device {self.device_rank} normalized mask: {mask}")

        loss = CalculateCrossEntropy(logits = logits, positives = mask)

        return loss

class RMSNorm(nn.Module):
    '''
    torch RMSNorm implements eps = 1e-6; setting default to 1e-5 in accordance with parameters used by DA paper and
    for consistency with LN (which torch uses eps = 1e-6)
    '''
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


