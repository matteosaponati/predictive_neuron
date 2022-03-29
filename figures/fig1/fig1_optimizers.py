"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_optimizers.py"
numerical comparison on simple optimization process with 2 pre-synaptic inputs.
optimizers: online, online with hard-bound, SGD (offline), Adam (offline)

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
par.T = 300
par.batch = 1
par.epochs = 600
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

'----------------'
def num_solution(par,neuron,x_data,online=False,bound=False):
    
    v,z = [], []
    
    for t in range(par.T):    
        v.append(neuron.v)      
        'online update'
        if online: 
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online(bound)    
        'update state variables'        
        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z

'----------------'
def train(par,neuron,x_data,online=False,bound=False):
    
    'allocate outputs'
    E_out = []
    w1, w2 = [], []
    v_out, spk_out = [],[]
    
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = num_solution(par,neuron,x_data,online,bound)
        'evaluate loss'
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x_data)
        if online == False:
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
        'output'
        E_out.append(E.item())
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        v_out.append(v)
        spk_out.append(z)
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return E_out, w1, w2, v_out, spk_out

'----------------'

'online optimization'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .03
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
'optimization'
E_on, w1_on, w2_on, v_on, spk_on = train(par,neuron,x_data,
                                      online=True)

'online optimization with hard bounds'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .03
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
'optimization'
E_onh, w1_onh, w2_onh, v_onh, spk_onh = train(par,neuron,x_data,
                                      online=True,bound='hard')

'offline optimization: BPTT with SGD'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .03
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
'optimization through bptt'
optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
E_sgd, w1_sgd, w2_sgd, v_sgd, spk_sgd = train(par,neuron,x_data)

'offline optimization: BPTT with Adam'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .03
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
'optimization through bptt'
optimizer = torch.optim.Adam(neuron.parameters(),
                              lr=1e-3,betas=(.9,.999))
E_adam, w1_adam, w2_adam, v_adam, spk_adam = train(par,neuron,x_data)

#%%

'plots'

fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(w1_on,color='mediumvioletred',linewidth=2,label = 'online')
plt.plot(w2_on,color='mediumvioletred',linewidth=2,linestyle='dashed')
plt.plot(w1_sgd,color='navy',linewidth=2,label = 'sgd')
plt.plot(w2_sgd,color='navy',linewidth=2,linestyle='dashed')
plt.plot(w1_adam,color='purple',linewidth=2,label = 'adam')
plt.plot(w2_adam,color='purple',linewidth=2,linestyle='dashed')
plt.plot(w1_onh,color='lightblue',linewidth=2,label = 'online - hard')
plt.plot(w2_onh,color='lightblue',linewidth=2,linestyle='dashed')
plt.xlabel(r'epochs')
plt.ylabel(r'synaptic weights $\vec{w}$')
plt.xlim(0,par.epochs)
plt.legend()
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])

fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(E_on,color='mediumvioletred',linewidth=2,label = 'online')
plt.plot(E_sgd,color='navy',linewidth=2,label = 'sgd')
plt.plot(E_adam,color='purple',linewidth=2,label = 'adam')
plt.plot(E_onh,color='lightblue',linewidth=2,label = 'online - hard')
plt.xlabel(r'epochs')
plt.ylabel(r'$\mathcal{L}$')
plt.xlim(0,par.epochs)
plt.legend()
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])

fig = plt.figure(figsize=(6,6), dpi=300)
for k,j in zip(spk_on,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
for k,j in zip(spk_sgd,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='navy',s=7)
for k,j in zip(spk_adam,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='purple',s=7)
for k,j in zip(spk_onh,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='lightblue',s=7)
plt.ylabel(r'output spikes (s) [ms]')
for k in timing*par.dt:
    plt.axhline(y=k,color='k',linewidth=1.5,linestyle='dashed')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,10)
plt.legend()
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    