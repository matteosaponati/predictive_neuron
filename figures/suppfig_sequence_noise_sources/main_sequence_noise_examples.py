"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequence_noise_examples.py":
Effect of different noise sources in the pre-synaptic sequence (Figure S4)

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
par.name = 'sequence'
par.dt = .05
par.eta = 5e-4
par.tau_m = 10.
par.v_th = 1.4
par.tau_x = 2.

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_seq = 100
par.N_dist = 100
par.N = par.N_seq+par.N_dist   
timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)

'set training algorithm'
par.bound = 'soft'
par.epochs = 1000

'set initialization'
par.init = 'fixed'
par.init_mean = 0.01
par.init_a, par.init_b = 0, .03

'set noise sources'
par.noise = 1
par.upload_data = 0
par.freq_noise = 1
par.freq = 10
par.jitter_noise = 1
par.jitter = 2
par.T = int(2*(par.Dt*par.N_seq + par.jitter)/par.dt) 
par.onset = 1
par.onset_list = np.random.randint(0,par.T/2,par.epochs)

'---------------------------------------------'
'no noise'
par.noise = 0
par.freq_noise = 0
par.jitter_noise = 0
par.onset = 0

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/spk_nonoise.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/spk_nonoise.pdf', format='pdf', dpi=300)
plt.close('all')

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
plt.savefig(os.getcwd()+'/w_nonoise.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/w_nonoise.pdf', format='pdf', dpi=300)
plt.close('all')


'---------------------------------------------'
'random onset'
par.noise = 1
par.freq_noise = 0
par.jitter_noise = 0
par.onset = 1

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/spk_onset.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/spk_onset.pdf', format='pdf', dpi=300)
plt.close('all')

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
plt.savefig(os.getcwd()+'/w_onset.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/w_onset.pdf', format='pdf', dpi=300)
plt.close('all')


'---------------------------------------------'
'jitter'
par.noise = 1
par.freq_noise = 0
par.jitter_noise = 1
par.onset = 0

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/spk_jitter.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/spk_jitter.pdf', format='pdf', dpi=300)
plt.close('all')

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
plt.savefig(os.getcwd()+'/w_jitter.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/w_jitter.pdf', format='pdf', dpi=300)
plt.close('all')


'---------------------------------------------'
'background frequency'
par.noise = 1
par.freq_noise = 1
par.jitter_noise = 0
par.onset = 0

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/spk_freq.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/spk_freq.pdf', format='pdf', dpi=300)
plt.close('all')

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
plt.savefig(os.getcwd()+'/w_freq.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/w_freq.pdf', format='pdf', dpi=300)
plt.close('all')