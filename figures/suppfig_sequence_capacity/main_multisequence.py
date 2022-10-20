"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig2_capacity.py":
    
    - numerical investigation of the capacity of the model
    - dependence on overlap between synapses
    - dependence on model parameters

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
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

from predictive_neuron import models, funs_train

savedir = '/gs/home/saponatim/'

'set model'
par = types.SimpleNamespace()
par.epochs = 100
par.device = 'cpu'
par.dt = .05
par.eta = 1e-3
par.tau_m = 15.
par.v_th = 1.5
par.tau_x = 2.

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_sub = 20
par.delay = 40
par.batch = 3
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]
timing = []
for b in range(par.batch):
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))

'set training algorithm'
par.optimizer = 'Adam'
par.bound = 'none'
par.epochs = 600

'set initialization'
par.init = 'fixed'
par.init_mean = 0.02
par.init_a, par.init_b = 0, .06

'set noise sources'
par.name = 'multisequence'
par.noise = False
par.freq_noise = False
par.freq = 5
par.jitter_noise = False
par.jitter = 2
par.T = int((par.Dt*par.N + par.jitter)/(par.dt))

'---------------------------------------------'

'fix seed'
np.random.seed(1992)

#%%
'set model'
neuron = models.NeuronClass(par)
neuron = funs_train.initialize_weights_PyTorch(par,neuron)

'training'
w,v,spk,loss = funs_train.train_PyTorch(par,neuron,timing=timing)

x = funs.get_multisequence(par,timing)
w,v,spk,loss = funs_train.train_PyTorch(par,neuron,x=x)

#%%
'plots'

'Pnale b'
spk_plot = [[] for b in range(par.batch)]
for b in range(par.batch):
    
    for e in range(par.epochs):
        spk_plot[b].append(spk[e][b])
        
    fig = plt.figure(figsize=(7,6), dpi=300)
    for k,j in zip(spk_plot[b],range(par.epochs)):
        plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
    plt.ylabel(r'time [ms]')
    plt.ylim(0,par.T*par.dt)
    plt.xlim(0,par.epochs)
    plt.xlabel('epochs')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(os.getcwd()+'/spk_seq_{}.png'.format(b), format='png', dpi=300)
    # plt.savefig(os.getcwd()+'/spk_seq_{}.pdf'.format(b), format='pdf', dpi=300)
    plt.close('all')

'Panel c'
w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N)
plt.imshow(np.fliplr(w_plot).T/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/w.png', format='png', dpi=300)
# plt.savefig(os.getcwd()+'/w.pdf', format='pdf', dpi=300)
plt.close('all')

#%%
w_plot = np.vstack(w)
plt.imshow(w_plot,aspect='auto')

#%%
b = 0

spk_plot = [[] for b in range(par.batch)]
for b in range(par.batch):
    for e in range(par.epochs):
        spk_plot[b].append(spk[e][b])

#%%
b = 0
for k,j in zip(spk_plot[b],range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')