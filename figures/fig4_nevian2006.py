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
par.tau_m = 25.
par.v_th = 2.
par.tau_x = 2.

par.T = int(300/par.dt)

'initial conditions'
par.init = 'fixed'
par.init_mean = 0.
w_0 = np.array([.01,.08])

"""
-------------------------------------------------------------------
we reproduce the experimental protocol by increasing the frequency of post bursts
inputs:
    1. n_spk: total number of post spikes in the bursts
    2. dt_burst, dt: delay between post spikes, delay between pre and first post
"""

burst_sweep = (np.array([10,20,50])/par.dt).astype(int)
dt = int(10/par.dt)
n_spikes = 3

w_prepost = np.zeros(burst_sweep.size)
w_postpre = np.zeros(burst_sweep.size)

for k, dt_burst in enumerate(burst_sweep):

    print(dt_burst)

    par.train_nb = par.batch
    par.test_nb = par.batch

    'pre-post pairing'

    spk_times = [np.array(0),dt+np.arange(0,dt_burst*n_spikes,dt_burst)]

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

    w_prepost[k] = neuron.w[0]

    'post-pre pairing'

    spk_times = [np.arange(0,dt_burst*n_spikes,dt_burst),
                 np.array(np.arange(0,dt_burst*n_spikes,dt_burst)[-1]+ dt)] 
    
    x = np.zeros((par.batch,par.N,par.T))
    for b in range(par.batch):
        for n in range(par.N):    
            if n < par.N_seq: x[b,n,spk_times[n]] = 1
            x[b,n,:] = np.convolve(x[b,n,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  

    neuron = NeuronClassNumPy(par)
    neuron.initialize()
    neuron.w = w_0.copy()[::-1]
    trainer = TrainerClass(par,neuron,x,x)

    for e in range (par.epochs):
        _ = trainer._do_train()

    w_postpre[k] = neuron.w[1] 

'-------------------------------'

'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(1e3/(burst_sweep[::-1]*par.dt),np.array(w_prepost)[::-1]/w_0[0],
         color='royalblue',linewidth=2,label='pre-post')
plt.plot(1e3/(burst_sweep[::-1]*par.dt),np.array(w_postpre)[::-1]/w_0[0],
         color='rebeccapurple',linewidth=2,label='post-pre')

'add experimental data'
x = [20,50,100]
y_pre, y_pre_e = [1.1,2,2.25],[.3,.3,.6]
y_post, y_post_e = [.74,.74,.55],[.2,.1,.15]
plt.scatter(x,y_pre,color='k',s=20)
plt.errorbar(x,y_pre,yerr = y_pre_e,color='k',linestyle='None')
plt.scatter(x,y_post,color='k',s=20)
plt.errorbar(x,y_post,yerr = y_post_e,color='k',linestyle='None')

fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.xlabel('frequency [Hz]')
plt.ylabel(r'$w/w_0$')
plt.savefig('plots/fig4_nevian2006.pdf', format='pdf', dpi=300)
plt.close('all')

"RMS error"
error_prepost = np.sqrt(((np.array(w_prepost)/w_0[0] - np.array(y_pre))**2).sum()/len(y_pre))
error_postpre = np.sqrt(((np.array(w_postpre)/w_0[0] - np.array(y_post))**2).sum()/len(y_post))

print("""
    Nevian et al (2006)

    RMS error:
     - prepost pairing: {}
     - postpre pairing: {}    

    """.format(error_prepost,error_postpre))