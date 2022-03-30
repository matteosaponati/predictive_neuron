"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig_timescales_separation.py":
    
    - numerical difference in synaptic weights depending on the
    voltage and learning timescale

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

from predictive_neuron import models, funs

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.T = 500
par.batch = 1
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 1e-5
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.
par.freq = 0

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data = funs.get_sequence(par,timing)

#%%

tau = np.linspace(1,50,40)
eta = np.logspace(-8,-1,40)

w1_on, w2_on = np.zeros((40,40)), np.zeros((40,40))
w1_sgd, w2_sgd = np.zeros((40,40)), np.zeros((40,40))

for k in range(40):
    
    par.eta = eta[k]
    for j in range(40):
        print('{} and {}'.format(j,k))
        
        par.tau_m = tau[j]
        
        '--------'

        neuron = models.NeuronClass(par)
        loss = nn.MSELoss(reduction='sum')
        w_0 = .01
        neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
        
        neuron.state()
        for t in range(par.T):       
            'online update'
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online()    
            'update state variables'        
            neuron(x_data[:,t])        
        
        w1_on[j,k] = neuron.w[0].item()
        w2_on[j,k] = neuron.w[1].item()
            
        '--------'    
        
        neuron = models.NeuronClass(par)
        loss = nn.MSELoss(reduction='sum')
        w_0 = .01
        neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
        optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
        
        v = []
        neuron.state()
        for t in range(par.T):  
            v.append(neuron.v)       
            neuron(x_data[:,t])    
            
        x_hat = torch.einsum("bt,j->btj",torch.stack(v,dim=1),neuron.w)
        E = .5*loss(x_hat,x_data)
        
        optimizer.zero_grad()
        E.backward()
        optimizer.step()
        
        w1_sgd[j,k] = neuron.w[0].item()
        w2_sgd[j,k] = neuron.w[1].item()
    
    
#%%
plt.yticks(range(40)[::5],tau[::5].round(0))
plt.xticks(range(40)[::10],eta[::10].round(7))

test = np.abs(w1_sgd-w1_on)/w1_sgd
test[test==0]=10e-8
#plt.imshow(np.abs(w1_sgd-w1_on),norm=matplotlib.colors.LogNorm());plt.colorbar()

plt.imshow(test,norm=matplotlib.colors.LogNorm());plt.colorbar()
        