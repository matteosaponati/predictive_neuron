import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

class NeuronClass():
    """ NeuronClass for a single predictive neuron (NumPy).

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

    def __init__(self, par):

        self.par = par  
        self.alpha = (1 - self.par.dt / self.par.tau_m)              

    def initialize(self):
        """
        supported initialization methods:
        - 'trunc_gauss': Initialize the weights using a truncated Gaussian distribution.
        - 'uniform': Initialize the weights using a uniform distribution.
        - 'fixed': Initialize the weights with a fixed value.

        Attributes:
        - self.w (np.ndarray): initial values of synaptic weights (self.par.N)
        """

        if self.par.init == 'trunc_gauss':
            a = (self.par.init_a - self.par.init_mean) / (1 / np.sqrt(self.par.N))
            b = (self.par.init_b - self.par.init_mean) / (1 / np.sqrt(self.par.N))
            self.w = truncnorm(a, b, loc = self.par.init_mean, scale = 1 / np.sqrt(self.par.N)).rvs(self.par.N)
        elif self.par.init == 'uniform':
            self.w = np.random.uniform(0, self.par.init_mean, self.par.N)
        elif self.par.init == 'fixed':
            self.w = self.par.init_mean * np.ones(self.par.N)
        
    def state(self):
        """ initialization of neuron states.

        Attributes:
        - self.v (np.ndarray): membrane voltage (self.par.batch,)
        - self.z (np.ndarray): output spike (self.par.batch,) 
        - self.p (np.ndarray): eligibiliy traces (self.par.batch, self.par.N)
        - self.epsilon (np.ndarray): prediction errors (self.par.batch, self.par.N)
        - self.E (np.ndarray): global signal (self.par.batch,)
        - self.grad (np.ndarray): the gradient (self.par.batch, self.par.N)
        """

        self.v = np.zeros(self.par.batch)
        self.z = np.zeros(self.par.batch)
        self.p = np.zeros((self.par.batch, self.par.N))
        self.epsilon = np.zeros((self.par.batch, self.par.N))
        self.E = np.zeros(self.par.batch)
        self.grad = np.zeros((self.par.batch, self.par.N))
    
    def __call__(self, x):
        """ update of state variables for neuronal dynamics

        -   (v) the neuron membrane voltage encoding the decaying memory of the
            cell. This is implemented by low-pass filtering the input currents 
            weighted by their respective synaptic strengths. A soft-reset 
            mechanism is added after each post-synapic spikes to reset the
            neuron memory. 
        -   (z) encodes the spiking activity of the cell.
        """

        self.v = self.alpha * self.v + np.einsum('ij,j->i', x, self.w) \
                 - self.par.v_th * self.z
        self.z.fill(0)
        self.z[self.v - self.par.v_th > 0] = 1
    
    def backward_online(self, x):
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

        self.epsilon = x - np.einsum('i,j->ij', self.v, self.w)
        self.E = np.einsum('ij,j->i', self.epsilon, self.w)
        self.grad = - np.einsum('i,ij->ij', self.v, self.epsilon) \
                    - np.einsum('i,ij->ij', self.E, self.p)
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
        
        grad = self.par.eta * self.grad.mean(axis = 0)

        if self.par.bound == 'soft':
            self.w = self.w - self.w * grad

        else:
            self.w = self.w - grad
            if self.par.bound == 'hard': np.maximum(self.w, 0, out = self.w)