"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"sequences_dmd_example.py":
reproduce the dynamics of anticipation and predictive plasticity of Figure 1
    
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
par.dt = .05
par.tau_x = 2

'set input'
par.name = 'sequence'
par.sequence = 'deterministic'
par.Dt = 2
par.N_seq = 10
par.N_dist = 0
par.N = par.N_seq+par.N_dist   
timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)


'set noise sources'
par.noise = 0
par.upload_data = 0
par.freq_noise = 0
par.freq = 10
par.jitter_noise = 0
par.jitter = 2
par.T = int(2*(par.Dt*par.N_seq + par.jitter)/par.dt) 
par.onset = 0
# par.onset_list = np.random.randint(0,par.T/2,par.epochs)

x_data = funs.get_sequence_NumPy(par,timing)




X1, X2 = x_data[:-1,:], x_data[1:,:]

u,s,vh = np.linalg.svd(X1)


A = (u.T@X2@vh).T@np.linalg.inv(s*np.eye(par.N_seq-1))



