"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforganization_example.py":
train the neural network model with nearest-neighbours connections (Figure 3)

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
par.device = 'cpu'
par.dt = .05
par.eta = 5e-7
par.tau_m = 20.
par.v_th = 2.7
par.tau_x = 2.
par.nn = 10
par.lateral = 2
par.is_rec = True

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 1
par.batch = 1
par.upload_data = False

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 2
par.delay = 4
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): 
        timing[n].append((spk_times+n*par.delay/par.dt).astype(int))
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)
        
'set training algorithm'
par.online = True
par.bound = 'none'
par.epochs = 300

'set initialization'
par.init = 'fixed'
par.init_mean = 0.02
par.init_a, par.init_b = 0, .02
par.w_0rec = .0003

'---------------------------------------------'

## MAKE DESCRIPTION HERE
"""
there are three sources of noise for each epoch:
    1. jitter of the spike times (random jitter between -par.jitter and +par.jitter)
    2. random background firing following an homogenenous Poisson process with rate
    distributione between 0 and par.freq 
    3. another subset of N_dist pre-synaptic neurons that fire randomly according
    to an homogenenous Poisson process with randomly distribuited rates between
    0 and par.freq
"""

'set model'
network = models.NetworkClass_SelfOrg_NumPy(par)
network = funs_train.initialization_weights_nn_NumPy(par,network)

'training'
w,v,spk = funs_train.train_nn_NumPy(par,network,timing=timing)