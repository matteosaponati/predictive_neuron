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
par.eta = 2e-4
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
par.v_th = 2.
par.tau_x = 2.

par.T = int(400/par.dt)

'initial conditions'
par.init = 'fixed'
par.init_mean = 0.
w_0 = np.array([.001,.11])

"""
-------------------------------------------------------------------
we reproduce the classical pre-post pairing protocol by changing 
the delay between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
    2. tau_sweep: different values of the membrane time constant
"""

delay = np.arange(2,60,1,dtype=int)
tau_sweep = np.array([10.,15.,20.])

w_prepost = np.zeros((tau_sweep.size,delay.size))
w_postpre = np.zeros((tau_sweep.size,delay.size))

for k,par.tau_m in enumerate(tau_sweep):

    print(par.tau_m)
    
    for j, par.Dt in enumerate(delay):

        print(par.Dt)

        spk_times = get_spike_times(par)
        x,_ = get_dataset_sequence(par,spk_times)

        par.train_nb = par.batch
        par.test_nb = par.batch
        
        'pre-post pairing'

        neuron = NeuronClassNumPy(par)
        neuron.initialize()
        neuron.w = w_0.copy()
        trainer = TrainerClass(par,neuron,x,x)

        for e in range (par.epochs):
            _ = trainer._do_train()

        w_prepost[k,j] = neuron.w[0]

        'post-pre pairing'

        neuron = NeuronClassNumPy(par)
        neuron.initialize()
        neuron.w = w_0.copy()[::-1]
        trainer = TrainerClass(par,neuron,x,x)

        for e in range (par.epochs):
            _ = trainer._do_train()

        w_postpre[k,j] = neuron.w[1] 

'-------------------------------'

'plot'

c = ['purple','royalblue','navy']
fig = plt.figure(figsize=(6,6), dpi=300)
for k in range(len(tau_sweep)):
    plt.plot(-delay[::-1],w_prepost[k][::-1]/w_0[0],linewidth=2,color=c[k])
    plt.plot(delay,w_postpre[k]/w_0[0],linewidth=2,color=c[k])
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('plots/fig4_classical.pdf', format='pdf', dpi=300)
plt.close('all')