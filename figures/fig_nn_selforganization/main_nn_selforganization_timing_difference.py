"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforganization_timing_difference.py":
difference in network spike times before and after training - Figure 3
    
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

from predictive_neuron import funs_train, funs

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
quantification of the difference between the network spike timing at the 
beginning and at the end of training

We define a novel NetworkClass where the forward pass does not contain 
the update step for the synaptic weights. We assign to the network the set of
synaptic weights of the first and last epoch, respectively.
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

'get synaptic weights'
w = np.load(os.getcwd()+'/w_nn.npy')

'set number of repetitions for different noise realizations'
rep = 100
dt = np.zeros((w.shape[0],rep))

for e in range(1867,w.shape[0]):
    
    for k in range(rep):
        
        x = funs.get_sequence_nn_selforg_NumPy(par,timing)
        network = NetworkClass_Forward(par)
        network.w = w[e,:]
        _,_,spk = funs_train.train_nn_NumPy(par,network,x=x)
        
        if spk[-1][-1] != [] and spk[0][0] != []:
            dt[e,k] = spk[-1][-1][0] - spk[0][0][0]
        else: dt[e,k] = dt[e,k-1] 
        
np.save(os.getcwd()+'/timing_difference.npy',dt)     
        
'---------------------------------------------'
'plots'

'panel d'

fig = plt.figure(figsize=(6,4), dpi=300)    
plt.plot(dt.mean(axis=1),linewidth=2,color='mediumvioletred')
plt.fill_between(range(w.shape[0]),dt.mean(axis=1)+dt.std(axis=1),
                 dt.mean(axis=1)-dt.std(axis=1),color='mediumvioletred',alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylabel(r'$\Delta t$ [ms]')
plt.xlabel(r'epochs')
plt.savefig(os.getcwd()+'/plots/timing_difference.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/timing_difference.pdf',format='pdf', dpi=300)
plt.close('all')