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

par.bound = 'none'
par.eta = 1e-5
par.batch = 1
par.epochs = 20 # # of pairings in the STDP protocol
    
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

"""
we qualitatively reproduce the experimental protocol by changing the initial
value of the synaptic weight while keeping the STDP protocol fixed
inputs:
    1. w0_sweep: the different value of synaptic strength
"""

spk_times = get_spike_times(par)
x,onsets = get_dataset_sequence(par,spk_times)

spk_times = (np.array([2.,6.])/par.dt).astype(int)
print(spk_times)
    
x = np.zeros((par.batch,par.N,par.T))
for b in range(par.batch):
    for n in range(par.N):    
        if n < par.N_seq: x[b,n,spk_times[n]] = 1
        x[b,n,:] = np.convolve(x[b,n,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  

par.train_nb = par.batch
par.test_nb = par.batch

w0_sweep = np.arange(.01,.06,.01)

w_prepost = np.zeros(w0_sweep.size)

for j, w0_pre in enumerate(w0_sweep):

    neuron = NeuronClassNumPy(par)
    neuron.initialize()
    neuron.w = np.array([w0_pre,.13])
    trainer = TrainerClass(par,neuron,x,x)

    for e in range (par.epochs):
        _ = trainer._do_train()

    w_prepost[j] = neuron.w[0]

'-------------------------------'

'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(w0_sweep,w_prepost/w0_sweep,linewidth=2,color='mediumvioletred')
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.xlabel(r'$w_0$')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.savefig('plots/figS10.pdf', format='pdf', dpi=300)
plt.close('all')