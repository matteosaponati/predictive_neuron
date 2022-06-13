"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig2_sequence_example.py":
    
    - example of input sequence
    - example of spike output
    - example fo weights dynamics

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

par.dir = '/mnt/hpx/departmentN4/Matteo/predictive_neuron/fig2_pattern/'
savedir = '/gs/home/saponatim/'

par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'architecture'
par.N = 500
par.T = int(200/par.dt)
par.T_pattern = int(100/par.dt)
par.freq_pattern = .01
par.seed = 1992
par.batch = 1
par.epochs = 3000
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .03

par.init = 'trunc_gauss'
par.init_mean = .03
par.init_a = 0.01
par.init_b = .05

par.freq = .01
par.jitter = 2

type = 'pattern'
w = np.load(par.dir+'w_{}.npy'.format(type))
loss = np.load(par.dir+'loss_{}.npy'.format(type))
v = np.load(par.dir+'v_{}.npy'.format(type))
spk = np.load(par.dir+'spk_{}.npy'.format(type),allow_pickle=True)
mask = np.load(par.dir+'mask.npy')
