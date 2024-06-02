import torch
import torch.nn as nn
        
class NeuronClass(nn.Module):
    """ NeuronClass for a single predictive neuron (PyTorch).

    Attributes:
    - par (NameSpace): list of parameters to configure the neuron
    - alpha (float32): costant for recursive dynamics

    Methods:
    - initialize(): initialize synaptic weights
    - state() : initialize of network states
    - __call__(): update of state variables
    - backward_online(): computes the online weight update 
    - update_online(): update synaptic weights
    """

    def __init__(self,par):
        super(NeuronClass,self).__init__() 
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
            self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
            torch.nn.init.trunc_normal_(self.w, mean = self.par.init_mean, std = 1 / torch.sqrt(self.par.N),
                                    a = self.par.init_a, b = self.par.init_b)
        elif self.par.init == 'uniform':
            self.w = nn.Parameter(self.par.init_mean*torch.rand(self.par.N))
        elif self.par.init == 'fixed':
            self.w = nn.Parameter(self.par.init_mean*torch.ones(self.par.N)).to(self.par.device)
        else:
            self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
            torch.nn.init.normal_(self.w, mean = 0., std = 1 / torch.sqrt(self.par.N))
        
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

        self.v = torch.zeros(self.par.batch, device = self.par.device)
        self.z = torch.zeros(self.par.batch, device = self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N, device = self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N, device = self.par.device)
        self.E = torch.zeros(self.par.batch, device = self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N, device = self.par.device)
        
    def __call__(self,x):
        """
        Parameters:
        x : Tensor of shape (self.par.batch, self.par.N)
        
        Updates:
        - self.v: Tensor of shape (self.par.batch,) 
        - self.z: Tensor of shape (self.par.batch,)
        """

        self.v = self.alpha*self.v + torch.matmul(x, self.w) - self.par.v_th * self.z.detach()
        self.z = (self.v > self.par.v_th).float()
        
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

        self.epsilon = x - self.v.unsqueeze(1) * self.w
        self.E = torch.matmul(self.epsilon, self.w)
        self.grad = - (self.v.unsqueeze(1) * self.epsilon + self.E.unsqueeze(1) * self.p)
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

        grad_mean = self.par.eta * torch.mean(self.grad, dim=0)
        if self.par.bound == 'soft':
            self.w = self.w - self.w * grad_mean
        else:
            self.w = self.w - grad_mean
            if self.par.bound == 'hard': self.w = nn.Parameter(torch.clamp(self.w, min=0))
