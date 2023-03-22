"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_symmetrical.py"
reproduce symmetrical STDP windows with predictive plasticity

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

from predictive_neuron import models, funs, funs_train

'---------------------------------------------'
def train_stdp(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, _, _, _ = funs_train.forward_NumPy(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print(e)        
    return w1, w2
'---------------------------------------------'

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 1e-5
par.tau_m = 25.
par.v_th = 2.5
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(600/par.dt)
par.epochs = 100

'initial conditions'
w_0 = np.array([.04,.09])

'create input pattern'
def get_sequence_stdp(par,timing):    
    x_data = np.zeros((par.N,par.T))
    for n in range(par.N):
        x_data[n,timing[n]]= 1
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]        
    return x_data
  
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

#%%
        
'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    for t in range(par.T):    
        v.append(neuron.v) 
        neuron(x_data[:,t])  
        
        if neuron.z != 0: z.append(t*par.dt)    
    return neuron, v, z


def train(par,neuron,x_data):
    w1, w2 = [], []
    spk_out = []
    v_out = []
    'training'
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)
        
        'output'
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        spk_out.append(z)
        v_out.append(v)
        if e%50 == 0: print(e)
        
    return w1, w2, v_out, spk_out
'---------------------------------------------'
#%%

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
"""

delay = (np.arange(0,300,50)/par.dt).astype(int)

w_prepost,w_postpre = [],[]
wtot = [[],[]]
spk_prepost,spk_postpre = [], []
for k in range(len(delay)):
    
    'set inputs'
    timing = np.array([0,0+ delay[k]]).astype(int)
    x_data = get_sequence_stdp(par,timing)
    
    'pre-post pairing'
    neuron = NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2,v,spk = train(par,neuron,x_data)
    spk_prepost.append(spk)
    w_prepost.append(w1[-1])
    wtot[0].append(w1)
    wtot[1].append(w2)
    
    'post-pre pairing'
    neuron = NeuronClass_NumPy(par)
    neuron.w = w_0[::-1].copy()
    w1,w2,v,spk = train(par,neuron,x_data)
    spk_postpre.append(spk)
    w_postpre.append(w2[-1])  
    
#%%
    
for k in range(len(delay)):
    plt.plot(wtot[0][k]/w_0[0],label='{}'.format(delay[k]*par.dt))
plt.legend()

#%%
'plot'
#fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(-delay[::-1]*par.dt,w_prepost[::-1]/w_0[0],linewidth=2)
plt.plot(delay*par.dt,w_postpre/w_0[0],linewidth=2)
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
#fig.tight_layout(rect=[0, 0.01, 1, 0.96])
#plt.savefig('stdp_window.png', format='png', dpi=300)
#plt.savefig('stdp_window.pdf', format='pdf', dpi=300)
#plt.close('all')