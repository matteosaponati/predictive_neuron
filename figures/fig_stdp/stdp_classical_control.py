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
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

#from predictive_neuron import models, funs

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


#%%

'create input pattern'
def get_sequence_stdp(par,timing):    
    x_data = np.zeros((par.N,par.T))
    for n in range(par.N):
        x_data[n,timing[n]]= 1
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]        
    return x_data
 
'----------------'
def forward(par,neuron,x_data,idx):
    
    v,z = [], []
    for t in range(par.T):    
        v.append(neuron.v) 
        neuron(x_data[:,t],idx)  
        
        if neuron.z != 0: z.append(t*par.dt)    
    return neuron, v, z


def train(par,neuron,x_data,idx):
    w1, w2 = [], []
    spk_out = []
    v_out = []
    'training'
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data,idx)
        
        'output'
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        spk_out.append(z)
        v_out.append(v)
        if e%10 == 0: print(e)
        
    return w1, w2, v_out, spk_out
'---------------------------------------------'

#%%
"""
we define the NeuronClass to fix the synaptic weight of the supra-threshold input
    - idx: the index of the synapses subject to plasticity 
"""
class NeuronClass_NumPy():

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
    
    def __call__(self,x,idx):
        
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon[idx] =  x[idx] - self.w[idx]*self.v
        self.grad[idx] = self.v*self.epsilon[idx] + np.dot(self.epsilon[idx],self.w[idx])*self.p[idx]
        if self.par.bound == 'soft':
            self.w[idx] = self.w[idx] + self.w[idx]*self.par.eta*self.grad[idx]
        elif self.par.bound == 'hard':
            self.w[idx] = self.w[idx] + self.par.eta*self.grad[idx]
            self.w[idx] = np.where(self.w[idx]<0,np.zeros_like(self.w[idx]),self.w[idx])
        else: self.w[idx] = self.w[idx] + self.par.eta*self.grad[idx]
        
        'update eligibility traces'
        self.p[idx] = self.alpha*self.p[idx] + x[idx]
        
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
    x_data = get_sequence_stdp(par,timing)
    
    'pre-post pairing'
    neuron = NeuronClass_NumPy(par)
    neuron.w = np.array([.001,.12])
    w1,w2,v,spk = train(par,neuron,x_data,0)
    spk_prepost.append(spk)
    w_prepost.append(w1[-1])
    
    'post-pre pairing'
    neuron = NeuronClass_NumPy(par)
    neuron.w = np.array([.12,.06])
    w1,w2,v,spk = train(par,neuron,x_data,1)
    spk_postpre.append(spk)
    w_postpre.append(w2[-1])  
  #%%      
'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(-delay[::-1]*par.dt,np.array(w_prepost[::-1])/.001,linewidth=2)
plt.plot(delay*par.dt,np.array(w_postpre)/.06,linewidth=2)
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('stdp_window.png', format='png', dpi=300)
plt.savefig('stdp_window.pdf', format='pdf', dpi=300)
plt.close('all')