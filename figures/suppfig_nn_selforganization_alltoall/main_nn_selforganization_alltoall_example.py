"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforganization_alltoall_example.py":
train the neural network model with learnable, all-to-all recurrent connections

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

from predictive_neuron import models, funs_train, funs

par = types.SimpleNamespace()

'set model'
par.dt = .05
par.eta = 1e-6
par.tau_m = 25.
par.v_th = 2.9
par.tau_x = 2.
par.nn = 8
par.is_rec = True

'set noise sources'
par.noise = False
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
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
        timing[n].append((spk_times+n*par.delay/par.dt).astype(int))

'set initialization and training algorithm'
par.init = 'fixed'
par.init_mean = 0.06
par.init_a, par.init_b = 0, .02
par.w_0rec = .0003

'set training algorithm'
par.bound = 'none'
par.epochs = 2000

'set noise sources'
par.T = int((par.nn*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

"""
quantification of the number of neurons that needs to be activate such that
the network can recall the whole sequence. We show that the number of neurons 
required for the recall decreases consistently across epochs. 

During each epoch we use *par.nn* pre-synaptic inputs 
"""

def get_sequence_nn_subseqs(par,timing):
    
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
        if n in range(par.subseq):
        
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

class NetworkClass_Forward():
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
        self.w = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
        
    def state(self):
        """initialization of network state"""

        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)

    def __call__(self,x):
        
        'create total input'
        x_tot = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
        self.z_out = self.beta*self.z_out + self.z
        for n in range(self.par.nn): 
            
            x_tot[:,n] = np.concatenate((x[:,n],np.append(np.delete(self.z_out,n,axis=0),[0],axis=0)),axis=0)  
                
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.sum(x_tot*self.w,axis=0) \
                 - self.par.v_th*self.z
        self.z = np.zeros(self.par.nn)
        self.z[self.v-self.par.v_th>0] = 1
        
'--------------------'

'a) train the network'

# network = models.NetworkClass_SelfOrg_AlltoAll(par)
# network = funs_train.initialization_weights_nn_AlltoAll(par,network)

# w,v,spk = funs_train.train_nn_NumPy(par,network,timing=timing)

'b) get weights across epochs'
w = np.load(os.getcwd()+'/Desktop/w_alltoall.npy')

'set model with forward pass only'
network = NetworkClass_Forward(par)

'---------------------------------------------'
'plots'

'Panel b'

'before'
par.subseq, par.epochs = 1, 1
x = get_sequence_nn_subseqs(par,timing)
network.w = w[0,:]

_,_,spk = funs_train.train_nn_NumPy(par,network,x=x)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel(r'neurons')
plt.savefig(os.getcwd()+'/plots/spk_before.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/spk_before.pdf',format='pdf', dpi=300)
plt.close('all') 
 
'learning'
x = funs.get_sequence_nn_selforg_NumPy(par,timing)
network.w = w[10,:]

_,_,spk = funs_train.train_nn_NumPy(par,network,x=x)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel(r'neurons')
plt.savefig(os.getcwd()+'/plots/spk_learning.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/spk_learning.pdf',format='pdf', dpi=300)
plt.close('all') 

'after'
par.subseq, par.epochs = 2, 1
x = get_sequence_nn_subseqs(par,timing)
network.w = w[-1,:]

_,_,spk = funs_train.train_nn_NumPy(par,network,x=x)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][-1],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel(r'neurons')
plt.savefig(os.getcwd()+'/plots/spk_after.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/spk_after.pdf',format='pdf', dpi=300)
plt.close('all') 

'after spontaneous'
par.subseq, par.epochs = 2, 1
x = get_sequence_nn_subseqs(par,timing)
network.w = w[-1,:]

_,_,spk = funs_train.train_nn_NumPy(par,network,x=x)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(spk[n][-1],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel(r'neurons')
plt.savefig(os.getcwd()+'/plots/spk_after_spontaneous.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/spk_after_spontaneous.pdf',format='pdf', dpi=300)
plt.close('all') 

'Panel c'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.TwoSlopeNorm(vmin=w[-1,:].min(),vcenter=0, vmax=w[-1,:].max())
plt.imshow(np.flipud(w[-1,:]),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.title(r'$\vec{w}$')
plt.ylabel(r'inputs')
plt.xlabel(r'neurons')
plt.savefig(os.getcwd()+'/plots/w_nn.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/w_nn.pdf',format='pdf', dpi=300)
plt.close('all')