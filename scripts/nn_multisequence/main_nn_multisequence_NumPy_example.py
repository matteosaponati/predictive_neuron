import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train_inhibition

#%%

par = types.SimpleNamespace()

'training algorithm'
par.bound = 'None'
par.init = 'uniform'
par.init_mean = 0.03
par.init_a, par.init_b = 0, .03
par.epochs = 100
par.batch = 2

'set input sequence'
par.N = 40
par.nn = 5
par.Dt = 2

'set noise sources'
par.noise = 0
par.upload_data = 0
par.freq_noise = 0
par.freq = 5
par.jitter_noise = 0
par.jitter = 1

'network model'
par.is_rec = 1
par.w0_rec = -0.03
par.dt = .05
par.eta = 1e-5
par.tau_m = 10.
par.v_th = 3.
par.tau_x = 2.

'set total length of simulation'
par.T = int(2*(par.Dt*par.N + par.jitter)/(par.dt))

'set timing'
#spk_times = []
#for b in range(par.batch):
#    times = (np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt).astype(int)
#    np.random.shuffle(times)
#    spk_times.append(times)
timing = [[] for n in range(par.nn)]
for n in range(par.nn):
    for b in range(par.batch): 
#        timing[n].append(spk_times[b])
        timing[n].append((np.linspace(par.Dt,
                                  par.Dt*par.N,par.N)/par.dt).astype(int))
        times = (np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt).astype(int)
        np.random.shuffle(times)
        timing[n].append(times)
    
    
x = funs.get_multisequence_nn_NumPy(par,timing)

#%%

network = funs_train_inhibition.initialize_nn_NumPy(par)

w,spk,loss = funs_train_inhibition.train_nn_NumPy(par,network,x=x)

#%%
w_plot = np.zeros((len(w),w[0][0].shape[0],par.nn))

for k in range(w_plot.shape[0]):
    for n in range(par.nn):
        w_plot[k,:,n] = w[k][0]
        
#%%

output_time = np.zeros((par.batch,par.nn,par.epochs))
output_net = np.zeros((par.batch,par.epochs))
for e in range(par.epochs):
    for b in range(par.batch):
        for n in range(par.nn):
            if spk[e][n][b] != []: 
                output_time[b,n,e] = spk[e][n][b][-1]
        
        output_net[b,e] = np.median(output_time[b,:,e])


plt.plot(output_net[1,:])
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
n = 2
plt.imshow(w_plot[:,:,n][:,np.argsort(timing[0])[0]].T,aspect='auto')
plt.colorbar()

#%%
n= 2

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
            if spk[e][n][b] != []: selectivity[e,n,b] = 1
            

            
#%%
            
plt.imshow(selectivity[-100,:],aspect='auto')

#%%
for n in range(par.nn):
    plt.plot(v[0][0,:,n])
    plt.plot(v[0][1,:,n])



