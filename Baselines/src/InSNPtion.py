# -*- coding: utf-8 -*-
"""
@author: James V. Talwar
Created on Tue May 19 17:09:42 2020; Re-factored on December 10, 2022 at 16:39 :16


InSNPtion.py builds a multi-task FC-FFN neural network. It is designed to take in 
relevant data and output unactivated outputs.  The reason for the unactivation is that it 
is assumed activation will be handled with sigmoid, softmax, etc. Therefore activations should
be accounted for in the training script when critera are defined for each label and losses are grouped
together before the backwards pass. This takes advantage of the fact that pytorch built in loss
functions take in unactivated outputs (e.g., nn.CrossEntropyLoss()). For cases when this is 
not true can always activate first relevant output and then take lossfunction. Model architecture allows 
for toggling activation parameters between ReLU, Mish, LeakyReLU, and GeLU.

#NOTE: Using a batch size of 1 will throw an error due to pytorch's batch normalization expectations
so ensure running with batchsize > 1. If want to handle batch size > 1 can adapt script to use instance norm 
in case when inputDimensions[0] == 1
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import logging
logging.getLogger().setLevel(logging.INFO)

#Define a function to apply weight transforms to initialize the model  
def Initialize_Weights(laya, default = "kaiming"):
    if isinstance(laya, nn.Linear) and default.lower() == "xavier":
        torch.nn.init.xavier_normal_(laya.weight.data)
        torch.nn.init.xavier_normal_(laya.bias.data.unsqueeze(0))
    elif isinstance(laya, nn.Linear) and default.lower() == "kaiming":
        #logging.info("Initializing with Kaiming...")
        torch.nn.init.kaiming_normal_(laya.weight)
  
    
def Mish(x):
    return (x*torch.tanh(F.softplus(x)))

class FCFFN(nn.Module): 
    '''
     INPUTS:
     - inputDimension: The number of SNPs (i.e., input feature dimension)
     - numLayers: The number of hidden layers in the network (i.e., before the multi-task learning application layers)
     - layerWidths: The widths of each layer (how many cells in each layer) #Type list 
     - dropout: Fraction of nodes to dropout for regularization (default 0.5)
     - multitaskOutputs: The number of neurons in the final layer for each task (type list) --> defaulting to [1,1] for each for now (e.g, assuming a binary classification task for both)
     - activation: The activation function employed by the model. Valid choices include ReLU (default), Mish, LeakyReLU, Linear (i.e., no activation function so network is purely linear), and GELU
     - ageInclusion: Boolean indicating whether age is employed in model or not.  If True, age is added at the penultimate layer.
    
     OUTPUT: A list of unactivated ouputs 
    '''
    
    def __init__(self, inputDimension: int, numLayers: int, layerWidths: list, multitaskOutputs: list, dropout: float = 0.5, activation: str = "ReLU", ageInclusion: bool = False):
        super(FCFFN, self).__init__() 
        self.activation = activation
        self.ageInclusion = ageInclusion
        if ageInclusion:
            logging.info("Age Inclusion set to {}".format(ageInclusion))
            
        self.getWide = [inputDimension] + layerWidths[:numLayers]
        self.howManyLayers = numLayers      
                          
        self.hidden_layers = nn.ModuleList() 
        self.mt_outputs = nn.ModuleList() #multi-task output layers
        
        for i in range(len(self.getWide) - 1):
            print("Adding linear layer from {} nodes to {} nodes...".format(self.getWide[i], self.getWide[i+1]))
            self.hidden_layers.append(InSNPtion_Layer(self.getWide[i],self.getWide[i+1], dropout, False, activation))
            
        #Add in multitask layers here
        print("Multi-task layers initializing...")
        for i in range(len(multitaskOutputs)):
            print("Adding multi-task linear layer from {} nodes to {} nodes".format(self.getWide[len(self.getWide)-1] + int(ageInclusion), multitaskOutputs[i]))
            self.mt_outputs.append(InSNPtion_Layer(self.getWide[len(self.getWide) - 1] + int(ageInclusion), multitaskOutputs[i], dropout, True, activation))
            
        
    def forward(self, x, **kwargs):
        if x.size()[0] <= 1:
            raise ValueError("Batch Dimension should be > 1. If single-instance sample training/testing is desired change batchnorm1d to instance norm.")
            
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x, self.activation, False)
        
        #If age is included concatenate it with hidden layer output as the final element:
        if self.ageInclusion:
            x = torch.cat([x, kwargs["ageBatch"]], dim = 1) #include z-scored age before projecting to final dimension. If desired can concatenate age earlier as well...
        
        #Multi-task learning 
        model_outputs  = []
        for el in self.mt_outputs:
            model_outputs.append(el(x, self.activation, True))   
                
        return model_outputs
        
#Layer implementation
class InSNPtion_Layer(nn.Module): 
    def __init__(self, input_dim, output_dim, dropout, itsTheFinalCountdown, layer_act): #(self, input, output, dropout, isThisTheFinalLayerForMultiTask, layerActivation)
        super(InSNPtion_Layer, self).__init__()
        layer_operations = []
        if itsTheFinalCountdown: #If final layer, don't want to apply batch norm or dropout
            layer_operations = [nn.Linear(input_dim, output_dim)]
        else:
            if layer_act == "Mish":
                layer_operations = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.Dropout(dropout)]
                
            elif layer_act == "GELU":
                print("Initializing activation f(x) as GELU...\n")
                layer_operations = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.Dropout(dropout), nn.GELU()]
                
            elif layer_act == "LeakyReLU":
                print("Initializing activation f(x) as Leaky ReLU...\n")
                layer_operations = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.Dropout(dropout), nn.LeakyReLU(inplace=True)]
                
            elif layer_act == "Linear":
                print("Initializing to high-compute linear algebra (i.e., no activation function between layers, no batch norm, no dropout)")
                layer_operations = [nn.Linear(input_dim, output_dim)]
            
            else:
                print("Initializing activation f(x) as ReLU")
                layer_operations = [nn.Linear(input_dim, output_dim), nn.BatchNorm1d(output_dim), nn.Dropout(dropout), nn.ReLU(inplace=True)]
            
        self.layer = nn.Sequential(*layer_operations)
        #logging.info("Initializing Weights...")
        self.layer.apply(Initialize_Weights)
        
    def forward(self, x, activation, finalLayer):
        if (activation == "Mish") and (finalLayer == False):
            return Mish(self.layer(x))
        else:
            return self.layer(x)
        
  