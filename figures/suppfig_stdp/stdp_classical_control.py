"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_classical_control.py"
reproduce classical anti-symmetrical STDP windows with predictive plasticity
+ synapse of supra-threshold input is frozen (not subject to plasticity)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import os
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import funs, funs_train

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
par.eta = 1.5e-4
par.tau_m = 10.
par.v_th = 3.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 60

'initial conditions'
w_0 = np.array([.001,.11])
 
"""
we define the NeuronClass to fix the synaptic weight of the supra-threshold input
    - idx: the index of the synapses subject to plasticity 
"""
class NeuronClass_NumPy():

    def __init__(self,par,idx):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = np.zeros(self.par.N)
        self.idx = idx
        
    def state(self):
        """initialization of neuron state"""
        
        self.v = 0
        self.z = 0
        self.p = np.zeros(self.par.N)
        self.epsilon = np.zeros(self.par.N)
        self.grad = np.zeros(self.par.N)
    
    def __call__(self,x):
        
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon[self.idx] =  x[self.idx] - self.w[self.idx]*self.v
        self.grad[self.idx] = self.v*self.epsilon[self.idx] + \
                                np.dot(self.epsilon[self.idx],self.w[self.idx])*self.p[self.idx]
        if self.par.bound == 'soft':
            self.w[self.idx] = self.w[self.idx] + self.w[self.idx]*self.par.eta*self.grad[self.idx]
        elif self.par.bound == 'hard':
            self.w[self.idx] = self.w[self.idx] + self.par.eta*self.grad[self.idx]
            self.w[self.idx] = np.where(self.w[self.idx]<0,
                                    np.zeros_like(self.w[self.idx]),self.w[self.idx])
        else: self.w[self.idx] = self.w[self.idx] + self.par.eta*self.grad[self.idx]
        
        'update eligibility traces'
        self.p[self.idx] = self.alpha*self.p[self.idx] + x[self.idx]
        
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.dot(x,self.w) 
        if self.v-self.par.v_th>0: 
            self.z = 1
            self.v = self.v - self.par.v_th
        else: self.z = 0

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
"""

delay = (np.arange(1,20,2)/par.dt).astype(int)

w_prepost,w_postpre = [],[]
spk_prepost,spk_postpre = [], []
for k in range(len(delay)):
    
    'set inputs'
    timing = np.array([0,0+ delay[k]]).astype(int)
    x_data = funs.get_sequence_stdp(par,timing)
    
    'pre-post pairing'
    neuron = NeuronClass_NumPy(par,0)
    neuron.w = np.array([.001,.12])
    w1,w2 = train_stdp(par,neuron,x_data)
    w_prepost.append(w1[-1])
    
    'post-pre pairing'
    neuron = NeuronClass_NumPy(par,1)
    neuron.w = np.array([.12,.06])
    w1,w2 = train_stdp(par,neuron,x_data)
    w_postpre.append(w2[-1])  
   
'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(-delay[::-1]*par.dt,np.array(w_prepost[::-1])/.001,color='royalblue',linewidth=2)
plt.plot(delay*par.dt,np.array(w_postpre)/.06,color='royalblue',linewidth=2)
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig(os.getcwd()+'/plots/stdp_window.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/stdp_window.pdf', format='pdf', dpi=300)
plt.close('all')