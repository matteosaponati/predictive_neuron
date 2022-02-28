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
        
        def surr_grad(v,gamma,v_th):
            return gamma*(1/(np.abs(v - v_th)+1.0)**2)   
        
        self.epsilon =  x - self.v*self.w
        self.grad = self.v*self.epsilon + torch.dot(self.epsilon,self.w)*self.p
        self.p = ((1-self.dt/self.tau) + surr_grad(self.v,self.gamma,self.v_th))*self.p + x
        
    def backward_offline(self,v,z,x):
        
        return grad

'------------------'

def train(par,online=True):
    """
    training
    input:
        1. par: set of parameters
        2. online: check for online computation of the gradient
    """

    loss = torch.linalg.norm()
    w = []

    neuron = NeuronClass(par)
    for e in range(par.epochs):
        
        neuron.state()
        for t in range(par.T):
            
            neuron(x[:,t])
            if online: neuron.backwar_online(x[:,t])
            
        E = loss(x - v*neuron.w)
        E.backward()
        w.append(neuron.w.item())
            
    return w