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

'------------------'

class NeuronClass(nn.Module):
    """
    NEURON MODEL
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        super(NeuronClass,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
    def state(self):
        """initialization of neuron state"""
        
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        
    def __call__(self,x):
        """recursive dynamics step, numerical solution"""
        
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        """
        online evaluation of the gradient:
            - compute the local prediction error epsilon
            - compute the local component of the gradient
            - update the pre-synaptic traces
        """
        
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.grad = -(self.v[:,None]*self.epsilon + \
                        (self.epsilon@self.w)[:,None]*self.p)
        self.p = self.alpha*self.p + x
        
    def backward_offline(self,v,x):
        """
        offline evaluation of the gradient
        """
        
        epsilon = x - torch.einsum("bt,j->btj",v,self.w)
        filters = torch.tensor([self.alpha**(x.shape[1]-i-1) 
                                for i in range(v.shape[1])]).float().view(1, 1, -1).to(self.par.device)
        p = F.conv1d(x.permute(0,2,1),filters.expand(self.par.N,-1,-1),
                         padding=x.shape[1],groups=self.par.N)[:,:,1:x.shape[1]+1]
        ## check how to compute this
        grad = v*epsilon + epsilon@self.w*p
        return grad
        
    def update_online(self,bound=False):
        """
        online update of parameters
        soft: apply soft lower-bound, update proportional to parameters
        hard: apply hard lower-bound, hard-coded positive parameters
        """
        if bound == 'soft':
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*(self.w*torch.mean(self.grad,dim=0)))
        if bound == 'hard':
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*torch.mean(self.grad,dim=0))
            self.w = nn.Parameter(torch.where(self.w<0,
                                       torch.zeros_like(self.w),self.w))
        else:
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*torch.mean(self.grad,dim=0))
    
    def update_stdp(self,x,z):
        
        ## compute difference between spk pre and spk post
        ## compute the update for each synapse
        ## compute the "gradient" for update
        
        return
        
    
'------------------'


