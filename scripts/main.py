import numpy as np
import torch
import types
import torch.nn as nn

from predictive_neuron import models, funs

par = types.SimpleNamespace()
'architecture'
par.N = 2
par.T = 200
par.batch = 1
par.epochs = 300
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 5e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.
par.freq = 0
'set inputs'
timing = np.array([2.,6.])/par.dt
inputs = funs.get_sequence(par,timing)

neuron = models.NeuronClass(par)

