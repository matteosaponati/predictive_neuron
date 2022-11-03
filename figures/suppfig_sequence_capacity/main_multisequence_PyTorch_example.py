"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_multisequence_PyTorch_example.py":
    
numerical investigation of the capacity of the model with non-overlapping sequences

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
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

'training algorithm'
par.optimizer = 'Adam'
par.epochs = 400

par.dtype = torch.float
par.device = "cpu"

'set initialization'
par.init = 'fixed'
par.init_mean = 0.04
par.init_a, par.init_b = 0, .06

'set input'
par.name = 'multisequence'
par.Dt = 2
    
'set noise sources'
par.noise = 0
par.upload_data = 0
par.freq_noise = 0
par.freq = 10.
par.jitter_noise = 0
par.jitter = 2
par.upload_data = 0
    
'set model'
par.dt = .05
par.eta = 1e-3
par.tau_x = 2.


"""
Figure S5d

Examples of the capacity of the model for multiple independent sequences. 
We calculated the capacity without any noise source (no spike jitter, no distractor
neurons, no background firing, no random onset of the sequence)
"""

'---------------------------------------------------'

"""
CASE 1

N = 100
N_sub = 5
"""

par.tau_m = 10.
par.v_th = .5

par.N_sub = 5
par.batch = 20
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]

'set total length of simulation and total input size'
par.N = par.batch*par.N_sub
par.T = int(2*(par.Dt*par.N_sub + par.jitter)/(par.dt))
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]

'set timing'
timing = []
for b in range(par.batch):
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))

'set model'
neuron = models.NeuronClass(par)
neuron = funs_train.initialize_weights_PyTorch(par,neuron)

'train'
x = funs.get_multisequence(par,timing)
w,v,spk,loss = funs_train.train_PyTorch(par,neuron,x=x)
w_plot = np.vstack(w)
    
'compute capacity of the model - same as in Figure S3'
alpha = 0
for k in range(par.batch):
    
    if w_plot[-1,par.N_subseq[k][0]:par.N_subseq[k][-1]].argmax() == 0 \
    and spk[-1][k] != [] \
    and spk[-1][k][-1] < timing[0][-1]*par.dt: 
        alpha +=1/par.batch
print(alpha)
    
'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.imshow(np.flipud(w_plot.T)/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/plots/w_Nsub_{}_N_{}_example.png'.format(par.N_sub,par.N), format='png', dpi=300)
plt.close('all')

'---------------------------------------------------'

"""
CASE 2

N = 100
N_sub = 20
"""

par.tau_m = 20.
par.v_th = 1.

par.N_sub = 20
par.batch = 5
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]

'set total length of simulation and total input size'
par.N = par.batch*par.N_sub
par.T = int(2*(par.Dt*par.N_sub + par.jitter)/(par.dt))
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]

'set timing'
timing = []
for b in range(par.batch):
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))

'set model'
neuron = models.NeuronClass(par)
neuron = funs_train.initialize_weights_PyTorch(par,neuron)

'train'
x = funs.get_multisequence(par,timing)
w,v,spk,loss = funs_train.train_PyTorch(par,neuron,x=x)
w_plot = np.vstack(w)
    
'compute capacity of the model - same as in Figure S3'
alpha = 0
for k in range(par.batch):
    
    if w_plot[-1,par.N_subseq[k][0]:par.N_subseq[k][-1]].argmax() == 0 \
    and spk[-1][k] != [] \
    and spk[-1][k][-1] < timing[0][-1]*par.dt: 
        alpha +=1/par.batch
print(alpha)
    
'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.imshow(np.flipud(w_plot.T)/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/plots/w_Nsub_{}_N_{}_example.png'.format(par.N_sub,par.N), format='png', dpi=300)
plt.close('all')

'---------------------------------------------------'

"""
CASE 2

N = 200
N_sub = 40
"""

par.tau_m = 20.
par.v_th = 1.

par.N_sub = 40
par.batch = 5
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]

'set total length of simulation and total input size'
par.N = par.batch*par.N_sub
par.T = int(2*(par.Dt*par.N_sub + par.jitter)/(par.dt))
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]

'set timing'
timing = []
for b in range(par.batch):
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))

'set model'
neuron = models.NeuronClass(par)
neuron = funs_train.initialize_weights_PyTorch(par,neuron)

'train'
x = funs.get_multisequence(par,timing)
w,v,spk,loss = funs_train.train_PyTorch(par,neuron,x=x)
w_plot = np.vstack(w)
    
'compute capacity of the model - same as in Figure S3'
alpha = 0
for k in range(par.batch):
    
    if w_plot[-1,par.N_subseq[k][0]:par.N_subseq[k][-1]].argmax() == 0 \
    and spk[-1][k] != [] \
    and spk[-1][k][-1] < timing[0][-1]*par.dt: 
        alpha +=1/par.batch
print(alpha)
    
'plots'
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.imshow(np.flipud(w_plot.T)/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(os.getcwd()+'/plots/w_Nsub_{}_N_{}_example.png'.format(par.N_sub,par.N), format='png', dpi=300)
plt.close('all')