"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"models.py"
single neuron and network models for predictive plasticity

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
        """recursive dynamics step"""
        
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        """
        online evaluation of the gradient:
            - compute the local prediction error 
            - compute the local component of the gradient
            - update the pre-synaptic traces
        """
        
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.grad = -(self.v[:,None]*self.epsilon + \
                        (self.epsilon@self.w)[:,None]*self.p)
        self.p = self.alpha*self.p + x
        
    def update_online(self):
        """
        online update of parameters
        soft: apply soft lower-bound, update proportional to parameters
        hard: apply hard lower-bound, hard-coded positive parameters
        """
        
        if self.par.bound == 'soft':
            self.w =  nn.Parameter(self.w - 
                                   self.w*(self.par.eta*torch.mean(self.grad,dim=0)))
        if self.par.bound == 'hard':
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*torch.mean(self.grad,dim=0))
            self.w = nn.Parameter(torch.where(self.w<0,
                                       torch.zeros_like(self.w),self.w))
        else:
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*torch.mean(self.grad,dim=0))
    
    def backward_offline(self,v,x):
        """offline evaluation of the gradient"""
        
        epsilon = x - torch.einsum("bt,j->btj",v,self.w)
        filters = torch.tensor([self.alpha**(x.shape[1]-i-1) 
                                for i in range(v.shape[1])]).float().view(1, 1, -1).to(self.par.device)
        p = F.conv1d(x.permute(0,2,1),filters.expand(self.par.N,-1,-1),
                         padding=x.shape[1],groups=self.par.N)[:,:,1:x.shape[1]+1]
        ## check how to compute this
        grad = v*epsilon + epsilon@self.w*p
        return grad

'------------------'

class NeuronClass_NumPy():
    """
    NEURON MODEL (Numpy version - online update)
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = np.zeros(self.par.N)
        
    def state(self):
        """initialization of neuron state"""
        
        self.v = 0
        self.z = 0
        self.p = np.zeros(self.par.N)
        self.epsilon = np.zeros(self.par.N)
        self.grad = np.zeros(self.par.N)
    
    def __call__(self,x):
        
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon =  x - self.w*self.v
        self.grad = self.v*self.epsilon + np.dot(self.epsilon,self.w)*self.p
        
        'soft: apply soft lower-bound, update proportional to parameters'
        'hard: apply hard lower-bound, hard-coded positive parameters'
        
        if self.par.bound == 'soft':
            self.w = self.w + self.w*self.par.eta*self.grad
        elif self.par.bound == 'hard':
            self.w = self.w + self.par.eta*self.grad
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w + self.par.eta*self.grad
        
        'update eligibility traces'
        self.p = self.alpha*self.p + x
        
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.dot(x,self.w) 
        if self.v-self.par.v_th>0: 
            self.z = 1
            self.v = self.v - self.par.v_th
        else: self.z = 0

'------------------'
'------------------'

class NetworkClass_SelfOrg(nn.Module):
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
        super(NetworkClass_SelfOrg,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        
        self.w = nn.Parameter(torch.empty((self.par.n_in+self.par.lateral,self.par.nn)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in+self.par.lateral))
        
    def state(self):
        """initialization of network state"""

        self.v = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        'external inputs + lateral connections'
        self.p = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)  

    def __call__(self,x):
        
        'create total input'
        x_tot = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        self.z_out = self.beta*self.z_out + self.z.detach()
        
        for n in range(self.par.nn):
            if n == 0:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([torch.zeros(self.par.batch,1),
                                                   self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1)   
            if n == self.par.nn-1:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                  torch.zeros(self.par.batch,1)],dim=1)],dim=1)   
            else:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                            torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                       self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1)        
        'update membrane voltages'
        for b in range(self.par.batch):
            self.v[b,:] = self.alpha*self.v[b,:] + torch.sum(x_tot[b,:]*self.w,dim=0) \
                     - self.par.v_th*self.z[b,:].detach()
        
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
        
        'create total input'
        x_tot = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        for n in range(self.par.nn):
            if n == 0:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([torch.zeros(self.par.batch,1),
                                                   self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1)   
            if n == self.par.nn-1:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                  torch.zeros(self.par.batch,1)],dim=1)],dim=1)   
            else:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                            torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                       self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1) 
        
        x_hat = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn)
        for b in range(self.par.batch):
            x_hat[b,:] = self.w*self.v[b,:]
            self.epsilon[b,:] = x_tot[b,:] - x_hat[b,:]
            self.grad[b,:] = -(self.v[b,:]*self.epsilon[b,:] \
                             + torch.sum(self.w*self.epsilon[b,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x_tot
        
    def update_online(self):
        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))

'------------------'
        
class NetworkClass_SelfOrg_NumPy():
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
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        self.w = np.zeros((self.par.n_in+self.par.lateral,self.par.nn))
        
    def state(self):
        """initialization of network state"""

        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)
        'external inputs + lateral connections'
        self.p = np.zeros((self.par.n_in+2,self.par.nn))
        self.epsilon = np.zeros((self.par.n_in+2,self.par.nn))
        self.grad = np.zeros((self.par.n_in+2,self.par.nn))  

    def __call__(self,x):
        
        'create total input'
        x_tot = np.zeros((self.par.n_in+2,self.par.nn))
        self.z_out = self.beta*self.z_out + self.z
        for n in range(self.par.nn): 
            if n == 0:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([0,self.z_out[n+1]])),axis=0)       
            elif n == self.par.nn-1:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],0])),axis=0)   
            else: 
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],self.z_out[n+1]])),axis=0) 
                
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon = x_tot - self.w*self.v
        self.grad = -(self.v*self.epsilon \
                         + np.sum(self.w*self.epsilon,axis=0)*self.p)
        self.p = self.alpha*self.p + x_tot
        
        'soft: apply soft lower-bound, update proportional to parameters'
        'hard: apply hard lower-bound, hard-coded positive parameters'
        
        if self.par.bound == 'soft':
            self.w = self.w + self.w*self.par.eta*self.grad
        elif self.par.bound == 'hard':
            self.w = self.w + self.par.eta*self.grad
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w + self.par.eta*self.grad
                
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.sum(x_tot*self.w,axis=0) \
                 - self.par.v_th*self.z
        self.z = np.zeros(self.par.nn)
        self.z[self.v-self.par.v_th>0] = 1
        
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
        
        x_hat = torch.zeros(self.par.batch,self.par.n_in,self.par.nn)
        for b in range(self.par.batch):
            x_hat[b,:] = self.w*self.v[b,:]
            self.epsilon[b,:] = x[b,:] - x_hat[b,:]
            self.grad[b,:] = -(self.v[b,:]*self.epsilon[b,:] \
                             + torch.sum(self.w*self.epsilon[b,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x
        
    def update_online(self):

        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))
