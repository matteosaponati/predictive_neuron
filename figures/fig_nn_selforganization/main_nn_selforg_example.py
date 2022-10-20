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
par.eta = 5e-5
par.tau_m = 15.
par.v_th = 4.
par.tau_x = 2.
par.nn = 8
par.lateral = 2
par.is_rec = True

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 5
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
par.init_mean = 0.03
par.init_a, par.init_b = 0, .06
par.w_0rec = .0003

'set training algorithm'
par.bound = 'none'
par.epochs = 500

'set noise sources'
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

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

#%%
plt.imshow(x[:,:,0],aspect='auto')


'set model'
network = models.NetworkClass_SelfOrg_NumPy(par)
network = funs_train.initialization_weights_nn_NumPy(par,network)
 
#%%


'training'
# w,v,spk = funs_train.train_nn_NumPy(par,network,x=x)
# 

w,v,spk = funs_train.train_nn_NumPy(par,network,timing=timing)


 #%%

m=1
for n in range(par.nn):
    plt.eventplot(spk[n][-5],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')

#%%

n = 6

for k in range(26):
    plt.plot(w[:,k,n],color='grey')
plt.plot(w[:,-2,n],color='red')
plt.plot(w[:,-1,n],color='blue')



#%%
plt.imshow(w[1,:,:])
plt.colorbar()


#%%

plt.plot(v[-1][-1,:])
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
'---------------------------------------------'
'plots'

'Panel c'   
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w[-1].min(),vcenter=0, vmax=w[-1].min())
plt.imshow(w[-1],cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel(r'inputs')
plt.xlabel(r'neurons')
plt.savefig('w_nn.png',format='png', dpi=300)
plt.savefig('w_nn.pdf',format='pdf', dpi=300)
plt.close('all')

'Panel d'
total_duration = np.zeros(par.epochs)
for e in range(par.epochs):
    if spk[-1][e] != [] and spk[0][e] != []:
        total_duration[e] = spk[-1][e][-1] - spk[0][e][0]
    else: total_duration[e-1]
fig = plt.figure(figsize=(6,6), dpi=300)    
plt.plot(total_duration,color='purple',linewidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel(r'$\Delta t$ [ms]')
plt.xlabel(r'epochs')
plt.savefig('total_duration.png',format='png', dpi=300)
plt.savefig('total_duration.pdf',format='pdf', dpi=300)
plt.close('all')    

















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

