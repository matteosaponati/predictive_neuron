import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train_inhibition

#%%

par = types.SimpleNamespace()

'training algorithm'
par.optimizer = 'Adam'
par.bound = 'None'
par.init = 'uniform'
par.init_mean = 0.02
par.init_a, par.init_b = 0, .03
par.epochs = 300
par.batch = 1
par.device = 'cpu'
par.dtype = torch.float

'set input sequence'
par.N = 50
par.nn = 10
par.Dt = 2

'set noise sources'
par.noise = 0
par.upload_data = 0
par.freq_noise = 0
par.freq = 5
par.jitter_noise = 0
par.jitter = 1

'network model'
par.is_rec = 0
par.w0_rec = 0.0
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'set total length of simulation'
par.T = int(2*(par.Dt*par.N + par.jitter)/(par.dt))

'set timing'
spk_times = []
for b in range(par.batch):
    times = (np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt).astype(int)
#    np.random.shuffle(times)
    spk_times.append(times)
timing = [[] for n in range(par.nn)]
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b])
#np.save(par.save_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}'.format(
#                    par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,par.rep),timing)
    
x = funs.get_multisequence_nn(par,timing)

#%%

network = funs_train_inhibition.initialize_nn(par)

w,v,spk,loss = funs_train_inhibition.train_nn(par,network,x=x)



#%%




'set model'
network = models.NetworkClass(par)
network = funs_train_inhibition.initialize_weights_nn_PyTorch(par,network)

x = funs.get_multisequence_nn(par,timing)

## check if training is really doing its job
#w, v, spk, loss = funs_train_inhibition.train_nn_PyTorch(par,network,x=x)
w, v, spk, loss = funs_train_inhibition.train_nn_PyTorch(par,network,timing=timing)

#%%

w_plot = np.zeros((len(w),w[0][0].shape[0],par.nn))

for k in range(w_plot.shape[0]):
    for n in range(par.nn):
        w_plot[k,:,n] = w[k][0]
    
    
#%%
    
plt.imshow(w_plot[:,:,0].T,aspect='auto')
plt.colorbar()

#%%
n= 4

plt.plot(w_plot[:,:,n])

#%%
n=0
plt.plot(v[-1])
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



