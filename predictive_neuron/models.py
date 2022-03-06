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
        par: model parameters
    """
    
    def __init__(self,par):
        super(NeuronClass,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        ## only positive values
        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
    def state(self):
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p, self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device), \
                                torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.grad = torch.zeros(self.par.N).to(self.par.device)
        
    def __call__(self,x):
        
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.grad = self.v[:,None]*self.epsilon + (self.epsilon@self.w)[:,None]*self.p
        self.p = self.alpha*self.p + x
        
    def backward_offline(self,v,z,x):
        
        epsilon = x - torch.einsum("btj,j->btj",v,self.w)
        filter = torch.tensor([(1-self.dt/self.tau)**(x.shape[1]-i-1) 
                                for i in range(v.shape[1])]).float()
        p = F.conv1d(x.permute(0,2,1),filter.expand(self.par.N,-1,-1),
                         padding=x.shape[1],groups=self.par.n)[:,:,1:x.shape[1]+1]
        grad = v*epsilon + epsilon@self.w*p
        return grad
    
'------------------'

def train(par,neuron,x_data,online=False):
    
    v = []
    for t in range(par.T):
        neuron(x_data[:,t])
        v.append(neuron.v)
        if online: neuron.backwar_online(x_data[:,t])
    
    return neuron, torch.stack(v,dim=1)

