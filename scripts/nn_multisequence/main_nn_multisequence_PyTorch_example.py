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
par.init = 'uniform'
par.init_mean = 0.05
par.init_a, par.init_b = 0, .03
par.epochs = 1000
par.batch = 2
par.device = 'cpu'
par.dtype = torch.float

'set input sequence'
par.n_in = 100
par.nn = 10
par.Dt = 2

'set noise sources'
par.noise = True
par.upload_data = False
par.freq_noise = True
par.freq = 5
par.jitter_noise = True
par.jitter = 1

'network model'
par.is_rec = True
par.w0_rec = -.05
par.dt = .05
par.eta = 1e-3
par.tau_m = 30.
par.v_th = 2.
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

x = funs.get_multisequence_nn(par,timing)

## check if training is really doing its job
#w, v, spk, loss = funs_train_inhibition.train_nn_PyTorch(par,network,x=x)
w, v, spk, loss = funs_train_inhibition.train_nn_PyTorch(par,network,timing=timing)

#%%

w_plot = np.zeros((len(w),w[0].shape[0],w[0].shape[1]))

for k in range(w_plot.shape[0]):
    w_plot[k] = w[k]
    
    
#%%
    
plt.imshow(w_plot[-1,:],aspect='auto')
plt.colorbar()

#%%
n= 4

plt.plot(w_plot[:,:,n])

#%%
n=4
plt.plot(v[-1][0,:,n])
plt.plot(v[-1][1,:,n])

#%%

selectivity = np.zeros((par.epochs,par.nn,par.batch))
for e in range(par.epochs):
    for n in range(par.nn):
        for b in range(par.batch):
            if spk[e][b][n] != []: selectivity[e,n,b] = 1
            
#%%
            
plt.imshow(selectivity[-100,:],aspect='auto')

#%%
for n in range(par.nn):
    plt.plot(v[0][0,:,n])
    plt.plot(v[0][1,:,n])



