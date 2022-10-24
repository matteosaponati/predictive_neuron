"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforganization_example.py":
train the neural network model with nearest-neighbours connections (Figure 3)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import os
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
par.eta = 5e-7
par.tau_m = 20.
par.v_th = 2.7
par.tau_x = 2.
par.nn = 10
par.lateral = 2
par.is_rec = True

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 1
par.batch = 1
par.upload_data = False

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 2
par.delay = 4
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): 
        timing[n].append((spk_times+n*par.delay/par.dt).astype(int))
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)
        
'set training algorithm'
par.online = True
par.bound = 'none'
par.epochs = 300

'set initialization'
par.init = 'fixed'
par.init_mean = 0.02
par.init_a, par.init_b = 0, .02
par.w_0rec = .0003

'---------------------------------------------'

## MAKE DESCRIPTION HERE
"""
there are three sources of noise for each epoch:
    1. jitter of the spike times (random jitter between -par.jitter and +par.jitter)
    2. random background firing following an homogenenous Poisson process with rate
    distributione between 0 and par.freq 
    3. another subset of N_dist pre-synaptic neurons that fire randomly according
    to an homogenenous Poisson process with randomly distribuited rates between
    0 and par.freq
"""

'set model'
network = models.NetworkClass_SelfOrg_NumPy(par)
network = funs_train.initialization_weights_nn_NumPy(par,network)

'training'
w,v,spk = funs_train.train_nn_NumPy(par,network,timing=timing)


'---------------------------------------------'
'plots'

"""
quantification of the number of neurons that needs to be activate such that
the network can recall the whole sequence. We show that the number of neurons 
required for the recall decreases consistently across epochs. 

During each epoch we use *par.nn* pre-synaptic inputs 
"""

def get_sequence_nn_selforg_NumPy(par,timing):
    
    'loop on neurons in the network'
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.freq_noise == True:
            prob = (np.random.randint(0,par.freq,par.n_in)*par.dt)/1000
            x = np.zeros((par.n_in,par.T))
            for nin in range(par.n_in): x[nin,:][np.random.rand(par.T)<prob[nin]] = 1        
        else:
            x = np.zeros((par.n_in,par.T))
        
        'span across neurons in the network'
        if n in par.subseq:
        
            'create sequence + jitter'
            if par.jitter_noise == True:
                timing_err = np.array(timing[n]) \
                              +  (np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt).astype(int)
                x[range(par.n_in),timing_err] = 1
            else: x[range(par.n_in),timing[n]] = 1
        
        'synaptic time constant'
        for nin in range(par.n_in):
            x[nin,:] = np.convolve(x[nin,:],
                          np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]   
            
        'add to total input'
        x_data.append(x)

    return np.stack(x_data,axis=2)

"""
Next, we use the set of synaptic inputs which the network learned when we showed
the whole input sequence. Consequently, we define a novel NetworkClass where the 
forward pass does not contain the update step for the synaptic weights. For each
training, we assign to the network the set of synaptic weights learned 
at the respective epoch.
"""

class NetworkClass_SelfOrg_NumPy():
    """
    NETWORK MODEL
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        self.w = np.zeros((self.par.n_in+self.par.lateral,self.par.nn))
        
    def state(self):
        """initialization of network state"""

        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)

    def __call__(self,x):
        
        'create total input'
        x_tot = np.zeros((self.par.n_in+2,self.par.nn))
        self.z_out = self.beta*self.z_out + self.z
        for n in range(self.par.nn): 
            if n == 0:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([0,self.z_out[n+1]])),axis=0)       
            elif n == self.par.nn-1:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],0])),axis=0)   
            else: 
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],self.z_out[n+1]])),axis=0) 
                
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.sum(x_tot*self.w,axis=0) \
                 - self.par.v_th*self.z
        self.z = np.zeros(self.par.nn)
        self.z[self.v-self.par.v_th>0] = 1
        
'------------------'

'Panel b'

'before'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_before.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_before.pdf',format='pdf', dpi=300)
plt.close('all') 
 
'learning'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][10],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_learning.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_learning.pdf',format='pdf', dpi=300)
plt.close('all') 

'after'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][-1],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_after.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_after.pdf',format='pdf', dpi=300)
plt.close('all') 

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