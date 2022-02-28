"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"models.py"
Predictive processes at the single neuron level 

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device

class NeuronClass(nn.Module):
    """
    NEURON MODEL
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and emission of output spike if voltage crosses the threshold
    inputs:
        1. par: model parameters
    """
    
    def __init__(self,par):
        super(NeuronClass,self).__init__() 
        
        self.par = par                
        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.device)
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
    def state(self):
        self.v = torch.zeros(self.par.batch)
        self.z = torch.zeros(self.par.batch)
        self.p, self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device), \
                                torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.grad = np.zeros(self.par.N).to(self.par.device)
        
    def __call__(self,x):
        
        self.v = (1-self.par.dt/self.par.tau)*self.v + np.dot(self.w,x) \
                    - self.par.v_th*self.z.detach()
        if self.v - self.par.v_th > 0: self.z = 1
        else: self.z = 0
        
    def backward_online(self,x):
        
        self.epsilon =  x - self.v*self.w
        self.grad = self.v*self.epsilon + torch.dot(self.epsilon,self.w)*self.p
        self.p = (1-self.dt/self.tau)*self.p + x
        
    def backward_offline(self,v,z,x):
        
        epsilon = x - v*self.w
        filter = torch.tensor([(1-self.dt/self.tau)**(x.shape[1]-i-1) 
                                for i in range(v.shape[1])]).float()
        p = F.conv1d(x.permute(0,2,1),filter.expand(self.par.N,-1,-1),
                         padding=x.shape[1],groups=self.par.n)[:,:,1:x.shape[1]+1]
        grad = v*epsilon + epsilon@self.w*p
        return grad

'------------------'

def train(par,online=True):

    loss = torch.linalg.norm()
    w = []

    neuron = NeuronClass(par)
    for e in range(par.epochs):
        
        ## set inputs
        x = funs.input()
        
        neuron.state()
        for t in range(par.T):
            
            neuron(x[:,t])
            if online: neuron.backwar_online(x[:,t])
            
        E = loss(x - neuron.v*neuron.w)
        E.backward()
        w.append(neuron.w.item())
            
    return w