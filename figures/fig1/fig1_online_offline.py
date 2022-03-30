"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_online_offine.py":
    
    - numerical comparison on simple optimization process 
     with 2 pre-synaptic inputs.
     
     - numerical difference in synaptic weights across epochs 
     and single-weight dynamics
     
     - dependence on the difference between membrane time constant 
     and learning rate

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

'set model'
par = types.SimpleNamespace()
par.N = 2
par.T = 300
par.batch = 1
par.epochs = 1000
par.device = 'cpu'
par.dt = .05
par.eta = 1e-5
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.
par.freq = 0

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data = funs.get_sequence(par,timing)

'initial condition'
w_0 = .03

'----------------'
def train(par,neuron,x_data,online=False,bound=False):
    
    'allocate outputs'
    E_out = []
    w1, w2 = [], []
    
    'training'
    for e in range(par.epochs):
        
        v = []    
        
        'forward pass'
        '-----------'
        neuron.state()
        for t in range(par.T): 
            v.append(neuron.v) 

            if online: 
                with torch.no_grad():
                    neuron.backward_online(x_data[:,t])
                    neuron.update_online(bound)  

            neuron(x_data[:,t])  
        '-----------'
        
        'evaluate loss'
        '-----------'
        x_hat = torch.einsum("bt,j->btj",torch.stack(v,dim=1),neuron.w)
        E = .5*loss(x_hat,x_data)
        if online == False:
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
        '-----------'
            
        'output'
        E_out.append(E.item())
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())

        if e%100 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return E_out, np.array(w1), np.array(w2)
'----------------'

'online optimization'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
E_on, w1_on, w2_on = train(par,neuron,x_data,
                                      online=True)

'offline optimization: BPTT with SGD'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
E_sgd, w1_sgd, w2_sgd = train(par,neuron,x_data)

'----------------'

'plots'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot((w1_on - w1_sgd),color='mediumvioletred',linewidth=2,label = r'd$w_1$')
plt.plot((w2_on - w2_sgd),color='navy',linewidth=2,label=r'd$w_2$')
plt.xlabel(r'epochs')
plt.ylabel(r'$\Delta \vec{w}$')
plt.xlim(0,par.epochs)
plt.legend()
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])

#%%
'---------------------'

neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
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
w_sgd = neuron.w.detach().numpy()

'---------------------'
'plots'

fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(w1,linewidth=2,color='mediumvioletred',label=r'$w_1$')
plt.axhline(y=w_sgd[0],color='mediumvioletred',linestyle='dashed')
plt.plot(w2,linewidth=2,color='navy',label=r'$w_2$')
plt.axhline(y=w_sgd[1],color='navy',linestyle='dashed')
plt.xlabel(r'epochs')
plt.ylabel(r'$\vec{w}$')
plt.legend()
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])


