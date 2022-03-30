"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_syn_dynamics.py"

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
par.T = 1000
par.batch = 1
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 1e-6
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.
par.freq = 0

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data = funs.get_sequence(par,timing)

'--------'

neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .1
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)

w1, w2 = [], []
neuron.state()
for t in range(par.T):       
    'online update'
    with torch.no_grad():
        neuron.backward_online(x_data[:,t])
        neuron.update_online()    
    'update state variables'        
    neuron(x_data[:,t])        

    w1.append(neuron.w[0].item())
    w2.append(neuron.w[1].item())
    
'--------'    

neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .1
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

w_fin = neuron.w.detach().numpy()


#%%

plt.plot(w1,linewidth=2)
plt.xlim(0,1000)
plt.axhline(y=w_fin[0],color='k',linestyle='dashed')


plt.plot(w2,linewidth=2)
plt.xlim(0,1000)
plt.axhline(y=w_fin[1],color='k',linestyle='dashed')




