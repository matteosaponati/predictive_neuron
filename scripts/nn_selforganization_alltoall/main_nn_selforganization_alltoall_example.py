"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforg_alltoall_example.py":
train the neural network model with learnable recurrent connections (Figure 3)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
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
par.device = 'cpu'
par.dt = .05
par.eta = 3e-6
par.tau_m = 25.
par.v_th = 2.9
par.tau_x = 2.
par.nn = 8
par.is_rec = True

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
## or jitter = 1
par.jitter = 2
par.batch = 1
par.upload_data = False

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 2
par.delay = 8
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): 
        # spk_times = (np.cumsum(np.random.randint(0,par.Dt,par.n_in))/par.dt).astype(int)
        timing[n].append((spk_times+n*par.delay/par.dt).astype(int))

'set initialization and training algorithm'
par.init = 'fixed'
par.init_mean = 0.06
par.init_a, par.init_b = 0, .02
par.w_0rec = .0003

'set training algorithm'
par.online = True
par.bound = 'none'
par.epochs = 25000

'set noise sources'
par.T = int((par.nn*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

'---------------------------------------------'

"""
there are three sources of noise for each epoch:
    1. jitter of the spike times (random jitter between -par.jitter and +par.jitter)
    2. random background firing following an homogenenous Poisson process with rate
    distributione between 0 and par.freq 
    3. another subset of N_dist pre-synaptic neurons that fire randomly according
    to an homogenenous Poisson process with randomly distribuited rates between
    0 and par.freq
"""
#%%
x = funs.get_sequence_nn_selforg_NumPy(par,timing)
plt.imshow(x[:,:,0],aspect='auto')

#%%

'set model'
network = models.NetworkClass_SelfOrg_AlltoAll(par)
network = funs_train.initialization_weights_nn_AlltoAll(par,network)

#%%

# x = funs.get_sequence_nn_selforg_NumPy(par,timing)
        
# w,v,spk = funs_train.train_nn_NumPy(par,network,x=x)

        
w,v,spk = funs_train.train_nn_NumPy(par,network,timing=timing)

#%%

plt.imshow(w[800])
plt.colorbar()

#%%

m=1
for n in range(par.nn):
    plt.eventplot(spk[n][2000],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
    
    for k in range(len(timing[n])+1):
        plt.axvline(x=timing[n][0][k]*par.dt,color='k')
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
#%%
n = 0
plt.plot(np.arange(0,par.T*par.dt,par.dt),v[-1][n,:])
for k in range(len(timing[n])+1):
        plt.axvline(x=timing[n][0][k]*par.dt,color='k')
# plt.xlim(0,20)
plt.ylim(0,par.v_th)


#%%

w_plot = np.zeros((par.epochs,par.n_in+par.nn,par.nn))
for k in range(par.epochs):
    
    w_plot[k,:,:] = w[k]

#%%

n = -1
for k in range(par.n_in):
    plt.plot(w_plot[:,k,n],color='grey')
for k in range(par.nn):
    plt.plot(w_plot[:,par.n_in+k,n],color='purple')

