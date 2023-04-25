import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

class NeuronClassNumPy():

    def __init__(self, par):
        """
        Args:
        - par: object containing the parameters of the neuron
        """
        self.par = par  
        self.alpha = (1 - self.par.dt / self.par.tau_m)              

    def initialize(self):
        """
        supported initialization methods:
        - 'trunc_gauss': Initialize the weights using a truncated Gaussian distribution.
        - 'uniform': Initialize the weights using a uniform distribution.
        - 'fixed': Initialize the weights with a fixed value.
        """
        if self.par.init == 'trunc_gauss':
            a = (self.par.init_a - self.par.init_mean) / (1 / np.sqrt(self.par.N))
            b = (self.par.init_b - self.par.init_mean) / (1 / np.sqrt(self.par.N))
            self.w = truncnorm(a, b, loc=self.par.init_mean, scale=1/np.sqrt(self.par.N)).rvs(self.par.N)
        elif self.par.init == 'uniform':
            self.w = np.random.uniform(0, self.par.init_mean, self.par.N)
        elif self.par.init == 'fixed':
            self.w = self.par.init_mean * np.ones(self.par.N)
        
    def state(self):
        """
        state variables:
        - self.v: Array of shape (self.par.batch,) representing the voltage.
        - self.z: Array of shape (self.par.batch,) representing the output spikes.
        - self.p: Array of shape (self.par.batch, self.par.N) representing the eligibiliy traces.
        - self.epsilon: Array of shape (self.par.batch, self.par.N) representing the prediction errors.
        - self.E: Array of shape (self.par.batch,) representing the global signal.
        - self.grad: Array of shape (self.par.batch, self.par.N) representing the gradient.
        """
        self.v = np.zeros(self.par.batch)
        self.z = np.zeros(self.par.batch)
        self.p = np.zeros((self.par.batch, self.par.N))
        self.epsilon = np.zeros((self.par.batch, self.par.N))
        self.E = np.zeros(self.par.batch)
        self.grad = np.zeros((self.par.batch, self.par.N))
    
    def __call__(self, x):
        """
        Args:
        x : Array of shape (self.par.batch, self.par.N)
        
        Updates:
        - self.v: Array of shape (self.par.batch,) 
        - self.z: Array of shape (self.par.batch,)
        """
        self.v = self.alpha * self.v + np.einsum('ij,j->i', x, self.w) \
                 - self.par.v_th * self.z
        self.z = np.zeros(self.par.batch)
        self.z[self.v - self.par.v_th > 0] = 1
    
    def backward_online(self, x):
        """
        Args:
        x : Array of shape (self.par.batch, self.par.N)
    
        Updates:
        - self.epsilon: Array of shape (self.par.batch, self.par.N) 
        - self.E: Array of shape (self.par.batch,) 
        - self.grad: Array of shape (self.par.batch, self.par.N) 
        - self.p: Array of shape (self.par.batch, self.par.N)
        """
        self.epsilon = x - np.einsum('i,j->ij', self.v, self.w)
        self.E = np.einsum('ij,j->i', self.epsilon, self.w)
        self.grad = - np.einsum('i,ij->ij', self.v, self.epsilon) \
                    - np.einsum('i,ij->ij', self.E, self.p)
        self.p = self.alpha * self.p + x
    
    def update_online(self):
        """
        'soft': the weight (w) is updated using a proportional term of 
        the gradient, scaled by the learning rate (eta) and the mean of 
        the gradient along the axis 0.

        'hard': the weight (w) is updated by subtracting the scaled gradient 
        (eta * grad.mean(axis=0)) from the weight (w). Then, any negative values 
        in the updated weight (w) are set to zero, enforcing a hard lowerbound 
        on weight values.

        'else':the weight (w) is updated by subtracting the scaled gradient.
        """
        if self.par.bound == 'soft':
            self.w = self.w - self.w * self.par.eta * self.grad.mean(axis=0)
        elif self.par.bound == 'hard':
            self.w = self.w - self.par.eta * self.grad.mean(axis=0)
            self.w = np.where(self.w < 0, np.zeros_like(self.w), self.w)
        else:
            self.w = self.w - self.par.eta * self.grad.mean(axis=0)

        
'---------------------------------------------------------------------------'

class NeuronClassPyTorch(nn.Module):

    def __init__(self,par):
        super(NeuronClassPyTorch,self).__init__() 
        """
        Args:
        - par: object containing the parameters of the neuron
        """
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              

    def initialize(self):
        """
        supported initialization methods:
        - 'trunc_gauss': Initialize the weights using a truncated Gaussian distribution.
        - 'uniform': Initialize the weights using a uniform distribution.
        - 'fixed': Initialize the weights with a fixed value.
        """
        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        if self.par.init == 'trunc_gauss':
            self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
            torch.nn.init.trunc_normal_(self.w, mean=self.par.init_mean, std=1/np.sqrt(self.par.N),
                                    a=self.par.init_a,b=self.par.init_b)
        if self.par.init == 'uniform':
            self.w = nn.Parameter(self.par.init_mean*torch.rand(self.par.N))
        if self.par.init == 'fixed':
            self.w = nn.Parameter(self.par.init_mean*torch.ones(self.par.N)).to(self.par.device)
        
    def state(self):
        """
        state variables:
        - self.v: Tensor of shape (self.par.batch,) representing the voltage.
        - self.z: Tensor of shape (self.par.batch,) representing the output spikes.
        - self.p: Tensor of shape (self.par.batch, self.par.N) representing the eligibiliy traces.
        - self.epsilon: Tensor of shape (self.par.batch, self.par.N) representing the prediction errors.
        - self.E: Tensor of shape (self.par.batch,) representing the global signal.
        - self.grad: Tensor of shape (self.par.batch, self.par.N) representing the gradient.
        """
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.E = torch.zeros(self.par.batch).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        
    def __call__(self,x):
        """
        Parameters:
        x : Tensor of shape (self.par.batch, self.par.N)
        
        Updates:
        - self.v: Tensor of shape (self.par.batch,) 
        - self.z: Tensor of shape (self.par.batch,)
        """
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        """
        Args:
        x : Tensor of shape (self.par.batch, self.par.N)
    
        Updates:
        - self.epsilon: Tensor of shape (self.par.batch, self.par.N) 
        - self.E: Tensor of shape (self.par.batch,) 
        - self.grad: Tensor of shape (self.par.batch, self.par.N) 
        - self.p: Tensor of shape (self.par.batch, self.par.N)
        """
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.E = self.epsilon@self.w
        self.grad = -(self.v[:,None]*self.epsilon + \
                        self.E[:,None]*self.p)
        self.p = self.alpha*self.p + x
        
    def update_online(self):
        """
        'soft': the weight (w) is updated using a proportional term of 
        the gradient, scaled by the learning rate (eta) and the mean of 
        the gradient along the axis 0.

        'hard': the weight (w) is updated by subtracting the scaled gradient 
        (eta * grad.mean(axis=0)) from the weight (w). Then, any negative values 
        in the updated weight (w) are set to zero, enforcing a hard lowerbound 
        on weight values.

        'else':the weight (w) is updated by subtracting the scaled gradient.
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
