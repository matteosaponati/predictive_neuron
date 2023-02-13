import numpy as np
import torch

class NeuronClassNumPy():
    
    def __init__(self,par):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = np.zeros(self.par.N)
        
    def state(self):
        
        self.v = 0
        self.z = 0
        self.p = np.zeros(self.par.N)
        self.epsilon = np.zeros(self.par.N)
        self.grad = np.zeros(self.par.N)
    
    def __call__(self,x):
        
        self.epsilon =  x - self.w*self.v
        self.grad = self.v*self.epsilon + np.dot(self.epsilon,self.w)*self.p
        
        if self.par.bound == 'soft':
            self.w = self.w + self.w*self.par.eta*self.grad
        elif self.par.bound == 'hard':
            self.w = self.w + self.par.eta*self.grad
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w + self.par.eta*self.grad
        
        self.p = self.alpha*self.p + x
        
        self.v = self.alpha*self.v + np.dot(x,self.w) 
        if self.v-self.par.v_th>0: 
            self.z = 1
            self.v = self.v - self.par.v_th
        else: self.z = 0

'---------------------------------------------------------------------------'

class NeuronClassPyTorch(nn.Module):

    def __init__(self,par):
        super(NeuronClassPyTorch,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
    def state(self):
        
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        
    def __call__(self,x):
        
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.grad = -(self.v[:,None]*self.epsilon + \
                        (self.epsilon@self.w)[:,None]*self.p)
        self.p = self.alpha*self.p + x
        
    ## maybe you don't want to update online also

    def update_online(self):
        
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
