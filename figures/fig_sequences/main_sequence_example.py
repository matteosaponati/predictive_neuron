"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"sequence_example_numpy.py":
train the single neuron model on high-dimensional input-spike trains (Figure 2)

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
par.device = 'cpu'
par.dt = .05
par.eta = 5e-4
par.tau_m = 10.
par.v_th = 1.5
par.tau_x = 2.
par.bound = 'soft'

'set input'
par.spk_volley = 'deterministic'
par.Dt = 2
par.N_seq = 10
par.N_dist = 10
par.N = par.N_seq+par.N_dist   
timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)

'set training algorithm'
par.bound = 'soft'
par.epochs = 100

'set initialization and training algorithm'
par.init = 'random'
par.init_mean = 0.2
par.init_a, par.init_b = 0, .4

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.T = int(2*(par.Dt*par.N_seq + par.jitter)/par.dt) 
par.onset = True
par.onset_list = np.random.randint(0,par.T/2,par.epochs)

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
neuron.w = funs_train.initialize_weights(par,neuron)

'training'
w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)

#%%
'---------------------------------------------'
'plots'

'Panel b'
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-par.onset_list[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'spike times $s$ [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')

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

'Panel c'
w = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N)
plt.imshow(w.T/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w.png', format='png', dpi=300)
plt.savefig('w.pdf', format='pdf', dpi=300)
plt.close('all')

ig = plt.figure(figsize=(1.5,5), dpi=300)
plt.ylim(0,par.N-1)
plt.xlim(-.5,.5)
plt.xticks([],[])
plt.imshow(w[:,0:10]/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w_conv.png', format='png', dpi=300)
plt.savefig('w_conv.pdf', format='pdf', dpi=300)
plt.close('all')