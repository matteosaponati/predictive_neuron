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
import scipy.stats as stats
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

from predictive_neuron import models, funs, funs_train

par = types.SimpleNamespace()

'set model'
par.device = 'cpu'
par.dt = .05
par.eta = 5e-4
par.tau_m = 10.
par.v_th = 1.5
par.tau_x = 2.
par.bound = 'soft'

'set noise sources'
par.onset = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.batch = 1

'set input'
par.spk_volley = 'deterministic'
par.Dt = 2
par.N_seq = 100
par.N_dist = 100
par.N = par.N_seq+par.N_dist   
# total length simulation is twice the length of the sequence
par.T = int(2*(par.Dt*par.N_seq + par.jitter)/(par.dt)) 
timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)

'set initialization and training algorithm'
par.init = 'random'
par.init_mean = 0.2
par.init_a, par.init_b = 0, .4

'set training algorithm'
par.seed = 1992
par.bound = 'soft'
par.epochs = 1000

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
np.random.seed(par.seed)
    
'set model'
neuron = models.NeuronClass_NumPy(par)
if par.init == 'fixed': 
    neuron.w = par.init_mean*np.ones(par.N)
if par.init == 'random':
    neuron.w = stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.N)), 
                          (par.init_b-par.init_mean)/(1/np.sqrt(par.N)), 
                          loc=par.init_mean, scale=1/np.sqrt(par.N)).rvs(par.N)

'training'
w_out = np.zeros((par.epochs,par.N))
spk_out = []
v_out = []
onset = []
loss_tot = []

for e in range(par.epochs):
    
    'get input data'
    onset.append(np.random.randint(0,par.T/2))
    x_data = funs.get_sequence_NumPy(par,timing,onset[e])
    
    'numerical solution'
    neuron.state()
    neuron, v, z , loss = funs_train.forward_NumPy(par,neuron,x_data)
    
    'output'
    w_out[e,:] = neuron.w.copy()
    spk_out.append(z)
    v_out.append(v)
    loss_tot.append(np.sum(loss))
    if e%50 == 0: print(e)


#%%
'compute metrics'

loss_out = np.array(loss_tot)
v_tot = np.zeros(par.epochs)
for e in range(par.epochs):
    v_tot[e] = np.sum(v_out[e])



#%%
'plot - panel c'
fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N)
plt.imshow(w_out.T/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w.png', format='png', dpi=300)
plt.savefig('w.pdf', format='pdf', dpi=300)
plt.close('all')

ig = plt.figure(figsize=(1.5,5), dpi=300)
plt.ylim(0,par.N-1)
plt.xlim(-.5,.5)
plt.xticks([],[])
plt.imshow(w_out[:,0:10]/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w_conv.png', format='png', dpi=300)
plt.savefig('w_conv.pdf', format='pdf', dpi=300)
plt.close('all')

#%%
'plot - panel b'
temp = spk_out.copy()
for k,j in zip(temp,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k)-onset[j]*par.dt,c='rebeccapurple',s=2)
plt.ylabel(r'spike times $s$ [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')

def fr(spk,T):
    return np.array([(len(spk[k])/T)*(1e3/T) for k in range(len(spk))])

fig,(ax1,ax2) = plt.subplots(2,1, sharex=True,figsize=(8,3))
ax1.bar(np.arange(0,epochs,1),fr,width=6,color='purple') 
ax2.bar(np.arange(0,epochs,1),fr,width=6,color='purple') 
ax1.set_ylim(np.max(fr)-1,np.max(fr)+1) 
ax2.set_ylim(0,np.max(fr[1:])) 
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False) 
ax2.xaxis.tick_bottom()
fig.subplots_adjust(hspace=0.05)
ax2.set_xlabel(r'epochs')
ax2.axhline(y=np.mean(fr), color='grey',linestyle='dashed',linewidth=1.5)
ax2.set_xlim(0,epochs)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'output_fr.png', format='png', dpi=300)
plt.savefig(savedir+'output_fr.pdf', format='pdf', dpi=300)
plt.savefig(savedir+'output_fr.svg', format='svg', dpi=300)
plt.close('all')

#%%
'plot - panel d'

loss_out = np.array(loss_tot)
fig = plt.figure(figsize=(4,6), dpi=300)
plt.xlabel(r'epochs')
plt.ylabel(r'$\mathcal{L}$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(loss_out/loss_out[0],'purple',linewidth=2)    
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('loss.png', format='png', dpi=300)
plt.savefig('loss.pdf', format='pdf', dpi=300)
plt.close('all')
    

fig = plt.figure(figsize=(4,6), dpi=300)
plt.xlabel(r'epochs')
plt.ylabel(r'$v_{tot}$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(v_tot/v_tot[0],'navy',linewidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('v_tot.png', format='png', dpi=300)
plt.savefig('v_tot.pdf', format='pdf', dpi=300)
plt.close('all')



        
#%%

# #%%

# plt.imshow(x_data,aspect='auto')
#%%


#%%

