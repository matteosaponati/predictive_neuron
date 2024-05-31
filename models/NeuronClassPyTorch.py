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

        self.par = par  
        self.alpha = (1 - self.par.dt / self.par.tau_m)              

    def initialize(self):
        """
        supported initialization methods:
        - 'trunc_gauss': Initialize the weights using a truncated Gaussian distribution.
        - 'uniform': Initialize the weights using a uniform distribution.
        - 'fixed': Initialize the weights with a fixed value.

        Attributes:
        - self.w (torch.Parameter): initial values of synaptic weights (self.par.N)
        """

        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
        torch.nn.init.normal_(self.w, mean = 0., std = 1 / torch.sqrt(self.par.N))
        
        if self.par.init == 'trunc_gauss':
            self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
            torch.nn.init.trunc_normal_(self.w, mean = self.par.init_mean, std = 1 / torch.sqrt(self.par.N),
                                    a = self.par.init_a, b = self.par.init_b)
        if self.par.init == 'uniform':
            self.w = nn.Parameter(self.par.init_mean * torch.rand(self.par.N))
        if self.par.init == 'fixed':
            self.w = nn.Parameter(self.par.init_mean * torch.ones(self.par.N)).to(self.par.device)
        
    def state(self):
        """ initialization of neuron states.

        Attributes:
        - self.v (torch.Tensor): membrane voltage (self.par.batch,)
        - self.z (torch.Tensor): output spike (self.par.batch,) 
        - self.p (torch.Tensor): eligibiliy traces (self.par.batch, self.par.N)
        - self.epsilon (torch.Tensor): prediction errors (self.par.batch, self.par.N)
        - self.E (torch.Tensor): global signal (self.par.batch,)
        - self.grad (torch.Tensor): the gradient (self.par.batch, self.par.N)
        """

        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch, self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch, self.par.N).to(self.par.device)
        self.E = torch.zeros(self.par.batch).to(self.par.device)
        self.grad = torch.zeros(self.par.batch, self.par.N).to(self.par.device)
        
    def __call__(self,x):
        """ update of state variables for neuronal dynamics

        -   (v) the neuron membrane voltage encoding the decaying memory of the
            cell. This is implemented by low-pass filtering the input currents 
            weighted by their respective synaptic strengths. A soft-reset 
            mechanism is added after each post-synapic spikes to reset the
            neuron memory. 
        -   (z) encodes the spiking activity of the cell.
        """

        self.v = self.alpha * self.v + x @ self.w \
                    - self.par.v_th*self.z.detach()
        
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        """ update of state variables for synaptic plasticity

        -   (epsilon) the prediction error computed as the difference between the 
            current input x and the reconstruction from the membrane voltage at 
            the previous timestep
        -   (E) the global error signal compute as the weighted sum of the prediction
            error of each synapses
        -   (grad) the current estimate of the gradient
        -   (p) the eligibility trace computed as a low-pass filtered version of the
            pre-synaptic activity
        """ 

        self.epsilon =  x - self.v[:,None] * self.w[None,:]
        self.E = self.epsilon @ self.w
        self.grad = - (self.v[:,None] * self.epsilon + self.E[:,None] * self.p)
        self.p = self.alpha * self.p + x
        
    def update_online(self):
        """ weight update

        -   (grad) the current estimate of the gradient, averaged across batches
            amd scaled by the learning rate

        -   'soft': the weight (w) is updated using a proportional term of 
            the gradient, scaled by the current value of the synaptic weight
        -   'hard': the weight (w) is updated by subtracting the scaled gradient 
            from the weight (w). Then, any negative values in the updated weight 
            are set to zero, enforcing a hard lowerbound on weight values
        """ 

        grad = self.par.eta * torch.mean(self.grad, dim=0)

        if self.par.bound == 'soft':
            w = self.w - self.w * grad
        else: 
            w = self.w - grad
            if self.par.bound == 'hard': w = torch.clamp(w, min = 0)

        self.w = nn.Parameter(w)

