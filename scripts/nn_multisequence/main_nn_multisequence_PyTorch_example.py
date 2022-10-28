import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train_inhibition

par = types.SimpleNamespace()

'training algorithm'
par.optimizer = 'Adam'
par.bound = 'None'
par.init = 'random'
par.init_mean = 0.1
par.init_a, par.init_b = 0, .03
par.epochs = 400
par.batch = 2
par.device = 'cpu'
par.dtype = torch.float

'set input sequence'
par.n_in = 4
par.nn = 3
par.Dt = 2

'set noise sources'
par.noise = True
par.upload_data = False
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2

'network model'
par.is_rec = True
par.w0_rec = -.05
par.dt = .05
par.eta = 5e-4
par.tau_m = 10.
par.v_th = 1.4
par.tau_x = 2.

'set total length of simulation'
par.T = int(2*(par.Dt*par.n_in + par.jitter)/(par.dt))

'set timing'
spk_times = []
for b in range(par.batch):
    times = (np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt).astype(int)
    np.random.shuffle(times)
    spk_times.append(times)
timing = [[] for n in range(par.nn)]
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b])
    
'set model'
network = models.NetworkClass(par)
network = funs_train_inhibition.initialize_weights_nn_PyTorch(par,network)

## check if training is really doing its job
w, v, spk, loss = funs_train_inhibition.train_nn_PyTorch(par,network,timing=timing)


