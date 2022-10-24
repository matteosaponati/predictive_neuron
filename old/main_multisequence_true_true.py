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
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train

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

# 'set inputs'
# par.Dt = 4
# par.batch = 4
# par.N_sub = 10
# par.N = int(par.batch*par.N_sub)
# par.T = int((par.N_sub*par.Dt)/par.dt)

# 'noise'
# par.offset = False
# par.freq_noise = False
# par.freq = .002
# par.jitter_noise = False
# par.jitter = 2
# par.seed = 1992
# par.init = 'fixed'
# par.w_0 = .03

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_sub = 10
par.delay = 40
par.batch = 4
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]
timing = []
for b in range(par.batch):
    # timing.append(((b*par.delay + np.cumsum(np.random.randint(0,par.Dt,par.N_sub)))/par.dt).astype(int))
    # timing.append((( np.cumsum(np.random.randint(0,par.Dt,par.N_sub)))/par.dt).astype(int))
    # timing.append(((b*par.delay + np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))

'set training algorithm'
par.bound = 'none'
par.epochs = 200

'set initialization and training algorithm'
par.init = 'fixed'
par.init_mean = 0.03
par.init_a, par.init_b = 0, .06

'set noise sources'
par.name = 'multisequence'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
# par.T = int((par.batch*par.delay + par.Dt*par.N + par.jitter)/(par.dt))
par.T = int((par.Dt*par.N + par.jitter)/(par.dt))


#%%

'fix seed'

'set model'
neuron = models.NeuronClass(par)
neuron = funs_train.initialize_weights_PyTorch(par,neuron)

x_data = funs.get_multisequence(par,timing)

#%%

plt.imshow(x_data[0,:,:].T,aspect='auto')

#%%

par.optimizer = 'Adam'

w,v,spk,loss = funs_train.train_PyTorch(par,neuron,timing=timing)

#%%
w_plot = torch.vstack(w,dim=0)


#%%
b = 0


spk_plot = [[] for b in range(par.batch)]
for b in range(par.batch):
    for e in range(par.epochs):
        spk_plot[b].append(spk[e][b])

#%%
b = 3
for k,j in zip(spk_plot[b],range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')