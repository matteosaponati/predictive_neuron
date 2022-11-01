"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"selforganization_replay.py"
compute how many neurons are needed for replay

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

from predictive_neuron import funs_train

par = types.SimpleNamespace()

'set model'
par.dt = .05
par.eta = 8e-7
par.tau_m = 25.
par.v_th = 3.
par.tau_x = 2
par.nn = 10
par.lateral = 2
par.is_rec = True

'set noise sources'
par.noise = 1
par.freq_noise = 1
par.freq = 10
par.jitter_noise = 1
par.jitter = 2
par.upload_data = 0

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 8
par.delay = 4
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): 
        timing[n].append((spk_times+n*par.delay/par.dt).astype(int))

par.epochs = 1

'set noise sources'
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

'---------------------------------------------'

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



'get weights across epochs'
w = np.load(os.getcwd()+'/w_nn.npy')

'count of the number of neurons required for full recall, across epochs'
'set number of repetitions for different noise realizations'
rep = 100
n_required = np.zeros((w.shape[0],rep))

count = 0
for e in range(w.shape[0]):
    
    for k in range(rep):
            
        'run across neurons in the network'
        for par.subseq in range(1,par.nn+1):
        
            'set model'
            network = NetworkClass_Forward(par)
            network.w = w[e,:].copy()
            
            'create input'
            x_subseq = get_sequence_nn_subseqs(par,timing)
                
            'training'
            _,_,spk = funs_train.train_nn_NumPy(par,network,x=x_subseq)
        
            '''
            span the spiking activity of every neuron in the network
            count if the neuron has been active during the simulaion
            '''
            count = 0
            for n in range(par.nn):
                if spk[n][0] != []: count+=1
            
            '''
            check if every neuron in the network was active during the simulation
            if true, the current value of *subseq* represent the amount of input
            needed for the network to recall the whole sequence.
            if false, the number of input presented to the network is not sufficient
            to trigger the recall of the whole sequence.
            '''
            if count == par.nn:
                n_required[e,k] = par.subseq
                break
            else:
                continue

'---------------------------------------'
'plot'

fig = plt.figure(figsize=(5,6), dpi=300)
plt.plot(n_required.mean(axis=1),linewidth=2,color='purple')
plt.fill_between(range(w.shape[0]),n_required.mean(axis=1)+n_required.std(axis=1),
                 n_required.mean(axis=1)-n_required.std(axis=1),
                 color='purple',alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel(r'# neurons for replay')
plt.savefig(os.getcwd()+'/plots/n_needed.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/n_needed.pdf',format='pdf', dpi=300)
plt.close('all')           