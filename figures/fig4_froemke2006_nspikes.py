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
par.eta = 8e-5
par.batch = 1
par.epochs = 30 # # of pairings in the STDP protocol
    
par.sequence = 'deterministic'
par.Dt = 4
par.N_seq = 2
par.N_dist = 0
par.N = par.N_seq+par.N_dist

par.freq = 0.
par.jitter = 0.
par.onset = 0

par.dt = .05
par.tau_m = 40.
par.v_th = 3.
par.tau_x = 2.

par.T = int(600/par.dt)

'initial conditions'
par.init = 'fixed'
par.init_mean = 0.
w_0 = np.array([.14,.017])

"""
-------------------------------------------------------------------
we reproduce the experimental protocol by increasing the number of inputs from
the second pre-synaptic neurons
input:
    
"""

n_spk_sweep = 5
dt_burst, dt = int(10/par.dt), int(5/par.dt)

w_postpre = np.zeros(n_spk_sweep)

for k, n_spk in enumerate(np.arange(1,n_spk_sweep+1)):

    print(n_spk)
    par.train_nb = par.batch
    par.test_nb = par.batch

    'post-pre pairing'

    spk_times = [(np.arange(0,10*n_spk,10)/par.dt).astype(int),dt]
    
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

'-------------------------------'

'plot'

fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(np.arange(1,n_spk+1),np.array(w_postpre)/w_0[1],color='rebeccapurple',linewidth=2)

'add experimental data'
x = [1,2,3,4,5]
y, y_e = [.7,.8,.9,1.02,1.2],[.1,.1,.1,.05,.05]
plt.scatter(x,y,color='k',s=20)
plt.errorbar(x,y,yerr = y_e,color='k',linestyle='None')
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.ylabel(r'$w/w_0$')
plt.xlabel(r'# spikes')

plt.xticks(np.arange(1,n_spk+1),np.arange(1,n_spk+1))
plt.ylim(.5,1.5)
plt.savefig('plots/fig4_froemke2006_nspikes.pdf', format='pdf', dpi=300)
plt.close('all')

"RMS error"
error = np.sqrt(np.sum((np.array(w_postpre)/w_0[1] - np.array(y))**2)/len(y))

print("""
    Froemke et al (2006)
    effect of # of post-syn spikes

    RMS error:
     - postpre pairing: {}    

    """.format(error))