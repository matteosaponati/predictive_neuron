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
        """initialization of neuron state variables"""
        
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
        """initialization of neuron state variables"""
        
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
        
        self.w = nn.Parameter(torch.empty((self.par.n_in+self.par.lateral,
                                           self.par.nn)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in+
                                                              self.par.lateral))
        
    def state(self):
        """initialization of network state variables"""

        self.v = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        'external inputs + lateral connections'
        self.p = torch.zeros(self.par.batch,self.par.n_in+self.par.lateral,
                             self.par.nn).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.n_in+self.par.lateral,
                                   self.par.nn).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.n_in+self.par.lateral,
                                self.par.nn).to(self.par.device)  

    def __call__(self,x):
        
        """
        - the external input x and the lateral connections from neuron n-1 and
        neuron n+1 constitute the the total input x_tot to neuron n
        - boundary conditions: neuron 0 and neuron par.nn only receive inputs from 
        neuron 1 and neuron par.nn-1, respectively
        """
        
        'convolve network output with synaptic time constant'
        self.z_out = self.beta*self.z_out + self.z.detach()
        'create total input x_tot'
        x_tot = torch.zeros(self.par.batch,self.par.n_in+self.par.lateral,
                            self.par.nn).to(self.par.device)
        for n in range(self.par.nn):
            if n == 0:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([torch.zeros(self.par.batch,1),
                                                   self.z_out.detach()[:,n+1].unsqueeze(1)],
                                                    dim=1)],dim=1)   
            if n == self.par.nn-1:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                  torch.zeros(self.par.batch,1)],dim=1)],
                                                    dim=1)   
            else:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                            torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                       self.z_out.detach()[:,n+1].unsqueeze(1)],
                                                    dim=1)],dim=1)        
    
    
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
                                                   self.z_out.detach()[:,n+1].unsqueeze(1)],
                                                    dim=1)],dim=1)   
            if n == self.par.nn-1:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                  torch.zeros(self.par.batch,1)],dim=1)],
                                                    dim=1)   
            else:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                            torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                       self.z_out.detach()[:,n+1].unsqueeze(1)],
                                                        dim=1)],dim=1) 
        
        'compute prediction error and update eligibility traces'
        x_hat = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn)
        for b in range(self.par.batch):
            x_hat[b,:] = self.w*self.v[b,:]
            self.epsilon[b,:] = x_tot[b,:] - x_hat[b,:]
            self.grad[b,:] = -(self.v[b,:]*self.epsilon[b,:] \
                             + torch.sum(self.w*self.epsilon[b,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x_tot
        
    def update_online(self):
        'update parameters'
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
        """initialization of network state variables"""

        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)
        'external inputs + lateral connections'
        self.p = np.zeros((self.par.n_in+2,self.par.nn))
        self.epsilon = np.zeros((self.par.n_in+2,self.par.nn))
        self.grad = np.zeros((self.par.n_in+2,self.par.nn))  

    def __call__(self,x):
        
        """
        - the external input x and the lateral connections from neuron n-1 and
        neuron n+1 constitute the the total input x_tot to neuron n
        - boundary conditions: neuron 0 and neuron par.nn only receive inputs from 
        neuron 1 and neuron par.nn-1, respectively
        """
        
        'convolve network output with synaptic time constant'
        self.z_out = self.beta*self.z_out + self.z
        'create total input x_tot'
        x_tot = np.zeros((self.par.n_in+2,self.par.nn))
        for n in range(self.par.nn): 
            if n == 0:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([0,self.z_out[n+1]])),axis=0)       
            elif n == self.par.nn-1:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],0])),axis=0)   
            else: 
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],self.z_out[n+1]])),axis=0) 
                
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon = x_tot - self.w*self.v
        self.grad = self.v*self.epsilon \
                         + np.sum(self.w*self.epsilon,axis=0)*self.p
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
        
class NetworkClass_SelfOrg_AlltoAll():
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
        self.w = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
        
    def state(self):
        """initialization of network state variables"""

        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)
        'external inputs + recurrent connections'
        self.p = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
        self.epsilon = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
        self.grad = np.zeros((self.par.n_in+self.par.nn,self.par.nn))

    def __call__(self,x):
        
        """
        the external input x and the internal activity of the network 
        constitute the the total input x_tot to neuron n
        """
        
        'create total input'
        x_tot = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
        self.z_out = self.beta*self.z_out + self.z
        for n in range(self.par.nn): 
            x_tot[:,n] = np.concatenate((x[:,n],np.append(np.delete(self.z_out,
                                             n,axis=0),[0],axis=0)),axis=0)  
                
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon = x_tot - self.w*self.v
        self.grad = self.v*self.epsilon \
                         + np.sum(self.w*self.epsilon,axis=0)*self.p
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

class NeuronClass_nn(nn.Module):
    """
     NEURON MODEL (nn implementation)
    - get the external and recurrent input vectors at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        super(NeuronClass_nn,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)
        self.beta = (1-self.par.dt/self.par.tau_x)  
        
        self.w = nn.Parameter(torch.empty((self.par.N)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
        if self.par.is_rec == 1:
            self.wrec = torch.empty((self.par.nn-1)).to(self.par.device)
            torch.nn.init.normal_(self.wrec, mean=0.0, std=1/np.sqrt(self.par.nn-1))
            
    def state(self):
        """initialization of neuron state variables"""
        
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        
    def __call__(self,x,x_rec=None):
        """recursive dynamics step"""
        
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        
        if self.par.is_rec == 1:
            'update the pre-synaptic input to the rest of the network'
            self.z_out = self.beta*self.z_out + self.z.detach()
            'update membrane voltage with recurrent dynamics'
            with torch.no_grad(): 
                self.v += x_rec@self.wrec
            
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

class NeuronClass_nn_NumPy(nn.Module):
    """
     NEURON MODEL (nn implementation)
    - get the external and recurrent input vectors at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        super(NeuronClass_nn_NumPy,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)
        self.beta = (1-self.par.dt/self.par.tau_x)  
        
        self.w = np.zeros(self.par.N)
        if self.par.is_rec == 1: self.wrec = np.zeros(self.par.nn-1)
            
    def state(self):
        """initialization of neuron state variables"""
        
        self.v = np.zeros(self.par.batch)
        self.z = np.zeros(self.par.batch)
        self.z_out = np.zeros(self.par.batch)
        self.p = np.zeros((self.par.batch,self.par.N))
        self.epsilon = np.zeros((self.par.batch,self.par.N))
        self.grad = np.zeros((self.par.batch,self.par.N))
 
    def __call__(self,x,x_rec=None):
        
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.grad = -(self.v[:,None]*self.epsilon + \
                        (self.epsilon@self.w)[:,None]*self.p)
        
        'update synaptic weights'
        if self.par.bound == 'soft':
            self.w =  self.w - self.w*(self.par.eta*np.mean(self.grad,axis=0))
        if self.par.bound == 'hard':
            self.w =  self.w - self.par.eta*np.mean(self.grad,axis=0)
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else:
            self.w =  self.w - self.par.eta*np.mean(self.grad,axis=0)
        
        'update eligibility traces'
        self.p = self.alpha*self.p + x
        
        'update membrane voltage'
        self.v = self.alpha*self.v + x@self.w - self.par.v_th*self.z
        if self.par.is_rec == 1:
            self.z_out = self.beta*self.z_out + self.z
            self.v += x_rec@self.wrec
            
        self.z = np.zeros(self.par.batch)
        self.z[self.v - self.par.v_th > 0] = 1