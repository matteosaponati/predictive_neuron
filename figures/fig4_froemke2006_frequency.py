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
par.eta = 3.4e-5
par.batch = 1
par.epochs = 40 # # of pairings in the STDP protocol
    
par.sequence = 'deterministic'
par.Dt = 4
par.N_seq = 2
par.N_dist = 0
par.N = par.N_seq+par.N_dist

par.freq = 0.
par.jitter = 0.
par.onset = 0

par.dt = .05
par.tau_m = 16.
par.v_th = 2.2
par.tau_x = 2.

par.T = int(500/par.dt)

'initial conditions'
par.init = 'fixed'
par.init_mean = 0.
w_0 = np.array([.12,.005])

"""
-------------------------------------------------------------------
we reproduce the experimental protocol by increasing the pairing frequency
inputs:
    1. dt_burst, dt: delay between pairing, delay between pre and post (in ms)
"""

burst_sweep = (np.array([100.,20.,10.])/par.dt).astype(int)
dt = int(6/par.dt)

w_postpre = np.zeros(burst_sweep.size)

for k, dt_burst in enumerate(burst_sweep):

    par.train_nb = par.batch
    par.test_nb = par.batch

    'post-pre pairing'

    spk_times = [np.arange(0,dt_burst*5,dt_burst),np.arange(0,dt_burst*5,dt_burst)+dt]
    
    x = np.zeros((par.batch,par.N,par.T))
    for b in range(par.batch):
        for n in range(par.N):    
            if n < par.N_seq: x[b,n,spk_times[n]] = 1
            x[b,n,:] = np.convolve(x[b,n,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  

    neuron = NeuronClassNumPy(par)
    neuron.initialize()
    neuron.w = w_0.copy()
    trainer = TrainerClass(par,neuron,x,x)

    for e in range (par.epochs):
        _ = trainer._do_train()

    w_postpre[k] = neuron.w[1] 

np.save('results/w_postpre_froemke2006_frequency',w_postpre)

'-------------------------------'

'plot'

fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.scatter(1e3/(np.array(burst_sweep)*par.dt),np.array(w_postpre)/w_0[1],color='rebeccapurple',s=40)
plt.plot(1e3/(np.array(burst_sweep)*par.dt),np.array(w_postpre)/w_0[1],color='rebeccapurple',linewidth=2)

'add experimental data'
x = [10,50,100]
y, y_e = [.7,.99,1.3],[.05,.05,.1]
plt.scatter(x,y,color='k',s=20)
plt.errorbar(x,y,yerr = y_e,color='k',linestyle='None')
fig.tight_layout(rect=[0, 0.01, 1, 0.96])

plt.xlabel('frequency [Hz]')
plt.ylabel(r'$w/w_0$')
plt.ylim(.5,1.5)
plt.savefig('plots/fig4_froemke2006_frequency.pdf', format='pdf', dpi=300)
plt.close('all')

"RMS error"
error = np.sqrt(np.sum((np.array(w_postpre)/w_0[1] - np.array(y))**2)/len(y))

print("""
    Froemke et al (2006)
    effect of pairing frequency

    RMS error:
     - postpre pairing: {}    

    """.format(error))