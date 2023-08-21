import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from models.NeuronClass import NeuronClassNumPy
from utils.TrainerClassNumPy import TrainerClass
from utils.data import get_spike_times, get_dataset_sequence

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'sequence'
par.package = 'NumPy'

par.bound = 'soft'
par.eta = 1.5e-4
par.batch = 1
par.epochs = 60 # # of pairings in the STDP protocol
    
par.sequence = 'deterministic'
par.Dt = 4
par.N_seq = 2
par.N_dist = 0
par.N = par.N_seq+par.N_dist

par.freq = 0.
par.jitter = 0.
par.onset = 0

par.dt = .05
par.tau_m = 10.
par.v_th = 3.
par.tau_x = 2.

par.T = int(200/par.dt)

'initial conditions'
par.init = 'fixed'
par.init_mean = 0.
w_0 = np.array([.001,.11])

'-----------------------------------------------------------------'

class NeuronClass(NeuronClassNumPy):
    
    def __init__(self,par,idx):
        super(NeuronClassNumPy, self).__init__()
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)       
        self.idx = idx     

    def backward_online(self,x):

        self.epsilon[:,self.idx] =  x[:,self.idx] - self.v*self.w[self.idx]
        self.E = self.epsilon[:,self.idx]*self.w[self.idx]
        self.grad = - self.v*self.epsilon[:,self.idx] \
                    - self.E*self.p[:,self.idx]
        self.p[:,self.idx] = self.alpha*self.p[:,self.idx] + x[:,self.idx]

'-----------------------------------------------------------------'

"""
-------------------------------------------------------------------
we reproduce the classical pre-post pairing protocol by changing 
the delay between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
"""

delay = np.arange(3,20,1,dtype=int)

w_prepost = np.zeros(delay.size)
w_postpre = np.zeros(delay.size)

for j, par.Dt in enumerate(delay):

    spk_times = get_spike_times(par)
    x,onsets = get_dataset_sequence(par,spk_times)

    par.train_nb = par.batch
    par.test_nb = par.batch
        
    'pre-post pairing'

    neuron = NeuronClass(par,0)
    neuron.initialize()
    neuron.w = np.array([.001,.12])
    trainer = TrainerClass(par,neuron,x,x)

    for e in range (par.epochs):
        _ = trainer._do_train()

    w_prepost[j] = neuron.w[0]

    'post-pre pairing'

    neuron = NeuronClass(par,1)
    neuron.initialize()
    neuron.w = np.array([.12,.06])
    trainer = TrainerClass(par,neuron,x,x)

    for e in range (par.epochs):
        _ = trainer._do_train()

    w_postpre[j] = neuron.w[1] 

'-------------------------------'

'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(-delay[::-1],w_prepost[::-1]/.001,linewidth=2,color='mediumvioletred')
plt.plot(delay,w_postpre/.06,linewidth=2,color='mediumvioletred')
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('plots/figS11.pdf', format='pdf', dpi=300)
plt.close('all')