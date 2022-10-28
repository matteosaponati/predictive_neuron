"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequences_capacity_example.py":
    
numerical investigation of the capacity of the model with non-overlapping sequences

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

par = types.SimpleNamespace()

'set model'
par.name = 'multisequence'
par.dt = .05
par.eta = 3e-7
par.tau_m = 18.
par.v_th = 2.6
par.tau_x = 2.

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_sub = 8
par.delay = 20
par.batch = 3
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]
timing = []
for b in range(par.batch):
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub) + 
                    b*(par.Dt*par.N_sub + par.delay))/par.dt).astype(int))

'set training algorithm'
par.bound = 'none'
par.epochs = 6000

'set initialization'
par.init = 'fixed'
par.init_mean = 0.04
par.init_a, par.init_b = 0, .06

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 5
par.jitter_noise = True
par.jitter = 2

par.T = int((par.Dt*par.N + (par.batch)*par.delay +par.jitter)/(par.dt))

'---------------------------------------------'

"""
there are two sources of noise for each epoch:
    1. jitter of the spike times (random jitter between -par.jitter and +par.jitter)
    2. random background firing following an homogenenous Poisson process with rate
    distributione between 0 and par.freq 
"""

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

'---------------------------------------------'
'plots'

'Panel b'
fig = plt.figure(figsize=(8,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/plots/spk_capacity.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/spk_capacity.pdf', format='pdf', dpi=300)
plt.close('all')

'Panel c'
w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N-1)
plt.imshow(w_plot.T/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/plots/w.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/w.pdf', format='pdf', dpi=300)
plt.close('all')