"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_multisequence_example.py":
train the single neuron model on multiple input-spike trains (Figure 2)

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

from predictive_neuron import models, funs_train

par = types.SimpleNamespace()

'set model'
par.name = 'multisequence'
par.device = 'cpu'
par.dt = .05
par.eta = 1e-6
par.tau_m = 10.
par.v_th = 1.2
par.tau_x = 2.

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_sub = 10
par.delay = 60
par.batch = 3
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]
timing = []
for b in range(par.batch):
    timing.append(((b*par.delay+np.linspace(
        par.Dt,par.Dt*par.N_sub,par.N_sub))/par.dt).astype(int))

'set training algorithm'
par.bound = 'none'
par.epochs = 500

'set initialization and training algorithm'
par.init = 'fixed'
par.init_mean = 0.03
par.init_a, par.init_b = 0, .06

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.T = int((par.batch*par.delay + par.Dt*par.N + par.jitter)/(par.dt))

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

'fix seed'
np.random.seed(1992)

'set model'
neuron = models.NeuronClass_NumPy(par)
neuron.w = funs_train.initialize_weights_NumPy(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

#%%
'---------------------------------------------'
'plots'

'Panel b'
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
for k in range(par.batch):
    plt.axhline(y=timing[k][-1]*par.dt,color='k',linestyle='dashed')
plt.ylabel(r'spike times $s$ [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')


#%%
fr = np.array([(len(spk[k])/par.T)*(1e3/par.T) for k in range(par.epochs)])
fig = plt.figure(figsize=(4,6), dpi=300)
plt.xlabel(r'epochs')
plt.ylabel(r'firing rate [Hz]')
plt.axhline(y=fr.mean(), color='black',linestyle='dashed',linewidth=1.5)
plt.plot(fr,'purple',linewidth=2)    
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('fr.png', format='png', dpi=300)
plt.savefig('fr.pdf', format='pdf', dpi=300)
plt.close('all')


#%%
'Panel c'
w = np.vstack(w)
# fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N)
plt.imshow(w.T,aspect='auto',cmap='coolwarm')#,norm=MidpointNormalize(midpoint=1))
plt.colorbar()
# fig.tight_layout(rect=[0, 0.01, 1, 0.97])
# plt.savefig('w.png', format='png', dpi=300)
# plt.savefig('w.pdf', format='pdf', dpi=300)
# plt.close('all')


#%%
'Panel d'





#%%
fig = plt.figure(figsize=(1.5,5), dpi=300)
plt.ylim(0,par.N-1)
plt.xlim(-.5,.5)
plt.xticks([],[])
plt.imshow(w[:,0:10]/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w_conv.png', format='png', dpi=300)
plt.savefig('w_conv.pdf', format='pdf', dpi=300)
plt.close('all')