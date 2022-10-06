"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig2_generalization.py":
    
    - numerical evaluation of generalization properties of neuron model
      (background firing, jitter in the sequence, distractors)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(neuron.v)              

        if par.optimizer == 'online':            
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online(par.hardbound)            

        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'-------------------'
def performance(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'create input data'
    timing = np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt
    if par.noise == 'background':
        x_data = funs.get_sequence_noise(par,timing,mu=True)
    if par.noise == 'jitter':
        x_data = funs.get_sequence_noise(par,timing,jitter=True)
    
    'set model'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
    'initialization with learned synaptic weights'
    w = np.load(par.dir+'w.npy')
    neuron.w = nn.Parameter(torch.tensor(w)).to(par.device)
    
    '----------------'
    'inference'
    neuron.state()
    neuron, v, z = forward(par,neuron,x_data)
    x_hat = torch.einsum("bt,j->btj",v,neuron.w)
    E = .5*loss(x_hat,x_data)
    '----------------'
    
    return E.item(), v, z
'-------------------'    

'-------------------'    
par = types.SimpleNamespace()

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'

'architecture'
par.N = 2
par.T = 300
par.batch = 1
par.epochs = 2000
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 3e-5
par.tau_m = 10.
par.v_th = 1.
par.tau_x = 2.

'inputs'
par.Dt = 2.
par.N = 100
par.T = int((2.+(par.Dt*par.N)+50) // par.dt)

'generalization'
par.noise = 'background'
par.freq = 5 
par.jitter = 2.

par.device = "cpu"
'-------------------'  

'-------------------'  

loss, v, spk = performance(par)





