"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforg_example.py":
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
par.eta = 1e-5
par.tau_m = 20.
par.v_th = 3.
par.tau_x = 2.
par.nn = 8
par.lateral = 2
par.is_rec = True
par.bound = 'none'

'set noise sources'
par.noise = False
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.batch = 1

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 26
par.delay = 4
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append((spk_times+n*par.delay/par.dt).astype(int))

'set initialization and training algorithm'
par.init = 'fixed'
par.init_mean = 0.08
par.init_a, par.init_b = 0, .12
par.w_0rec = .003

'set training algorithm'
par.bound = 'soft'
par.epochs = 10

'set noise sources'
par.noise = False
par.freq_noise = True
par.freq = 1
par.jitter_noise = True
par.jitter = 1
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

#%%
x = funs.get_sequence_nn_selforg_NumPy(par,timing)

'set model'
network = models.NetworkClass_SelfOrg_NumPy(par)
network = funs_train.initialization_weights_nn_NumPy(par,network)
 
#%%


'training'
w,v,spk = funs_train.train_nn_NumPy(par,network,x=x)


#%%

plt.imshow(w[-1])
plt.colorbar()
#%%

w = np.vstack(w)
# fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.imshow(w.T,aspect='auto',cmap='coolwarm')#,norm=MidpointNormalize(midpoint=1))
plt.colorbar()


#%%
# ## ADDING SET TIMING IN THE MAIN SCRIPT
# 'set timing'

# 'create timing'
# if par.random==True:
#     timing = [[] for n in range(par.nn)]
#     for n in range(par.nn):
#         for b in range(par.batch): 
#             spk_times = np.random.randint(0,(par.Dt/par.dt)*par.n_in,size=par.n_in)
#             timing[n].append(spk_times+n*(par.n_in*par.Dt/par.dt)+ par.delay/par.dt)
# else: 
#     timing = [[] for n in range(par.nn)]
#     spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
#     for n in range(par.nn):
#         for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt) #*(par.n_in*par.Dt/par.dt)+ 
        


'---------------------------------------------'

"""
"""

# 'fix seed'
# np.random.seed(par.seed)
    
'set model'
network = models.NetworkClass_SelfOrg_NumPy(par)
network.w = funs_train.initialization_weights_nn_NumPy(par,network)

'training'
w,v,spk = funs_train.train_nn_PyTorch(network)

