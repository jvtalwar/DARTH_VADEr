'''
@author: James V. Talwar
Created on: November 25, 2022

About: This python script offers a preposterous panoply of feature extraction functions and can be conceptualized as 
a feature extraction utility file, with methods called accordingly. Release your inner deep-learning archaelogist and 
enable excellent extreme excavatation. 

'''
import torch 
import torch.nn as nn
import logging
from typing import Dict, Iterable, Callable

#Define relevant classes and functions for feature extraction
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, keyWord: str):
        super().__init__()
        self.model = model
        layers = sorted([k for k in dict([*model.named_modules()]) if keyWord in k])
        logging.info("{} model layers identified with key word {}".format(len(layers), keyWord))
        self.features = {layer: torch.empty(0) for layer in layers}
        self.handles = dict() 

        for layerID in layers:
            layer = dict([*self.model.named_modules()])[layerID]
            handle = layer.register_forward_hook(self.SaveOutputHook(layerID))
            self.handles[layerID] = handle
            
    def SaveOutputHook(self, layerID: str) -> Callable:
        def fn(laya, weValueYourInput, output): #laya = layer (e.g. Linear(...); weValueYourInput = input tensor
            self.features[layerID] = output
        return fn

    def forward(self, x, **kwargs) -> Dict[str, torch.Tensor]:
        preds = self.model(x, **kwargs)
        return self.features, self.handles, preds