"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"models_nn.py"
Predictive processes at the network level

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

'------------------'

class NetworkClass(nn.Module):
    """
    NETWORK MODEL
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        super(NetworkClass,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        
        self.w = nn.Parameter(torch.empty((self.par.n_in,self.par.nn)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in))
        
        if self.par.is_rec == 'True':
            w_rec = np.random.randn(self.par.nn,self.par.nn)/np.sqrt(self.par.nn)
            w_rec = np.where(np.eye(self.par.nn)>0,np.zeros_like(w_rec),w_rec)
            self.wrec = nn.Parameter(torch.as_tensor(w_rec,dtype=self.par.dtype).to(self.par.device))
        
    def state(self):
        """initialization of network state"""
        
        self.v = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        
        self.p = torch.zeros(self.par.batch,self.par.n_in,self.par.nn).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.n_in,self.par.nn).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.n_in,self.par.nn).to(self.par.device)    
        
    def __call__(self,x):
        
        'update membrane voltages'
        for b in range(self.par.batch):
            self.v[b,:] = self.alpha*self.v[b,:] + torch.sum(x[b,:]*self.w,dim=0) \
                     - self.par.v_th*self.z[b,:].detach()
                     
        if self.par.is_rec == True: 
            self.z_out = self.beta*self.z_out + self.z.detach()
            self.v += self.z_out.detach()@self.wrec
        
        'update output spikes'
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z[self.v-self.par.v_th>0] = 1
        
    def backward_online(self,x):
        """
        online evaluation of the gradient:
            - compute the local prediction error 
            - compute the local component of the gradient
            - update the pre-synaptic traces
        """
        
        self.epsilon = x - self.w.unsqueeze(0)*self.v
        self.grad = -(self.v*self.epsilon + torch.sum(self.w.unsqueeze(0).repeat(self.par.batch,1,1)*self.epsilon,dim=1)*self.p)
        self.p = self.alpha*self.p + x
        
    def update_online(self):

        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))
        
'------------------'

def pseudo_derivative(v):
    gamma = 40
    return v/(1+gamma*torch.abs(v))

class surr_grad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input)
        out[input > 0] = 1.0
        return out
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input*pseudo_derivative(input)
        return grad
spk_fn  = surr_grad.apply

class StandardNetworkClass(nn.Module):

    def __init__(self,par,is_rec=False):
        super(StandardNetworkClass,self).__init__() 
        
        self.par = par
        self.device, self.dtype = par.device, par.dtype
        self.is_rec = is_rec
        self.w = nn.Parameter(torch.empty((self.par.n_in,self.par.n)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in))
            
        if is_rec:
            w_rec = np.random.randn(self.par.n,self.par.n)/np.sqrt(self.par.n)
            w_rec = np.where(np.eye(self.par.n)>0,np.zeros_like(w_rec),w_rec)
            self.wrec = nn.Parameter(torch.as_tensor(w_rec,dtype=self.dtype).to(self.par.device))
    
    def state(self):
        
        self.v = torch.zeros(self.par.N,self.par.n).to(self.par.device)
        self.z = torch.zeros(self.par.N,self.par.n).to(self.par.device)
                  
    def __call__(self,x):
        
        self.v = self.par.alpha*self.v + x@self.w - self.par.v_th*self.z.detach()
        if self.is_rec: self.v += self.z@self.wrec
        v_thr = self.v - self.par.v_th
        self.z = spk_fn(v_thr) 
    
'------------------'
    
class ReadoutClass(nn.Module):
    
    def __init__(self,par):
        super(ReadoutClass,self).__init__()
        self.par = par
        self.device, self.dtype = par.device, par.dtype
        self.w = nn.Parameter(torch.empty((self.par.n,self.par.n_out)).to(par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n))
    
    def state(self):
        self.y = torch.zeros((self.par.N,self.par.n_out)).to(self.par.device)
        
    def __call__(self,x):
        self.y = self.par.alpha*self.y + x@self.w
            
'------------------'