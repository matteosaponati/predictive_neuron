"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_membrane_dendrite.py"
reproduce qualitatively the dependence on membrane voltage accessible to the synapses 
and the switch from LTP to LTD given by distance

'Sjöström et al (2001) Rate, timing, and cooperativity jointly determine cortical 
synaptic plasticity. Neuron' (Figure 5C)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import funs

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 1e-4
par.tau_m = 20.
par.v_th = 2.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 40

'---------------------'
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
    
    def __init__(self,par,factor=1):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = np.zeros(self.par.N)
        self.factor = factor
        
    def state(self):
        """initialization of neuron state"""
        
        self.v, self.vfactor = 0, 0
        self.z = 0
        self.p = np.zeros(self.par.N)
        self.epsilon = np.zeros(self.par.N)
        self.grad = np.zeros(self.par.N)
    
    def __call__(self,x):
        
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon =  x - self.w*self.vfactor
        self.grad = (self.vfactor)*self.epsilon + np.dot(self.epsilon,self.w)*self.p
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
        self.vfactor = self.alpha*self.vfactor + np.dot(self.w,x)
        
        if self.v-self.par.v_th > 0:
            self.v = self.v - self.par.v_th
            self.vfactor = self.vfactor - self.factor*self.par.v_th
            self.z = 1
        else: self.z = 0
'---------------------'

"""
we qualitatively reproduce the experimental protocol by scaling the amount of
membrane potential available at the synaptic level
inputs:
    1. factor: scaling factor to decrease the depolarization level at synapses
"""

w_0 = np.array([.005,.1])
factor = np.arange(0,1.1,.1)

'set inputs'
timing = (np.array([2.,8.])/par.dt).astype(int)
x_data = funs.get_sequence_stdp(par,timing)

w_pre = []
for k in factor:
    
    'pre-post pairing'
    neuron = NeuronClass_NumPy(par,factor=k)
    neuron.w = w_0.copy()
    w1,w2 = funs.train(par,neuron,x_data)
    w_pre.append(w1[-1])
    
plt.plot(factor,np.array(w_pre)/w_0[0])
