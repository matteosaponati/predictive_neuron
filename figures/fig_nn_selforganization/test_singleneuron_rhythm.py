"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequence_example.py":
train the single neuron model on high-dimensional input-spike trains (Figure 2)

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

from predictive_neuron import models, funs_train, funs

par = types.SimpleNamespace()

'set model'
par.name = 'rhythms'
par.dt = .05
par.eta = 1e-4
par.tau_m = 10.
par.v_th = 5.
par.tau_x = 2.

'set input'
par.sequence = 'deterministic'
par.Dt = 100
par.N_seq = 50
par.N_dist = 50
par.N = par.N_seq+par.N_dist   
par.cycles = 2
spk_times = (np.linspace(par.Dt,par.Dt*par.cycles,par.cycles)/par.dt).astype(int)
timing = [[] for n in range(par.N_seq)]
for k in range(len(timing)):
    timing[k].append(spk_times)

'set training algorithm'
par.bound = 'soft'
par.epochs = 100

'set initialization'
par.init = 'random'
par.init_mean = 0.01
par.init_a, par.init_b = 0, .02

'set noise sources'
par.noise = True
par.cycle_prob = .4
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 4
par.T = int((par.Dt*(par.cycles+1) + par.jitter)/par.dt) 
par.onset = False
par.onset_list = np.random.randint(0,par.T/2,par.epochs)


#%%
x_data = funs.get_rhythms_NumPy(par,timing,onset=None)

#%%

plt.imshow(x_data,aspect='auto')
#%%
'---------------------------------------------'

"""
there are three sources of noise for each epoch:
    1. jitter of the spike times (random jitter between -par.jitter and +par.jitter)
    2. random background firing following an homogenenous Poisson process with rate
    distributione between 0 and par.freq 
    3. another subset of N_dist pre-synaptic neurons that fire randomly according
    to an homogenenous Poisson process with randomly distribuited rates between
    0 and par.freq
"""

'fix seed'
np.random.seed(1992)

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

w_plot = np.vstack(w)
plt.imshow(np.flipud(w_plot.T),aspect='auto')
plt.colorbar()

#%%

for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')


#%%

n = 2
plt.plot(np.arange(0,par.T*par.dt,par.dt),v[0])
for k in range(len(timing[n])+1):
        plt.axvline(x=timing[n][0][k]*par.dt,color='k')
# plt.xlim(0,20)
plt.ylim(0,par.v_th)

#%%

'---------------------------------------------'
'plots'

'Panel b'
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/spk.png', format='png', dpi=300)
# plt.savefig(os.getcwd()+'/spk.pdf', format='pdf', dpi=300)
plt.close('all')

fr = np.array([(len(spk[k])/par.T)*(1e3/par.T) for k in range(par.epochs)])
fig = plt.figure(figsize=(4,6), dpi=300)
plt.xlabel(r'epochs')
plt.ylabel(r'firing rate [Hz]')
plt.axhline(y=fr.mean(), color='black',linestyle='dashed',linewidth=1.5)
plt.plot(fr,'purple',linewidth=2)    
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/fr.png', format='png', dpi=300)
# plt.savefig(os.getcwd()+'/fr.pdf', format='pdf', dpi=300)
plt.close('all')

'Panel c'
w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N)
plt.imshow(w_plot.T/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/w.png', format='png', dpi=300)
# plt.savefig(os.getcwd()+'/w.pdf', format='pdf', dpi=300)
plt.close('all')