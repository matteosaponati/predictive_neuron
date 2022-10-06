"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_.py":
    
    - numerical comparison on simple optimization process 
    with 2 pre-synaptic inputs.
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
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

savedir = '/gs/home/saponatim/'

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.T = 300
par.batch = 1
par.epochs = 300
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 3e-5
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data,_ = funs.get_sequence(par,timing)

'----------------'
def forward(par,neuron,x_data,online=False,bound=False):
    
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
        neuron, v, z = forward(par,neuron,x_data,online,bound)
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
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return E_out, w1, w2, v_out, spk_out

'----------------'

#%%

w_0 = [torch.tensor([.1,.1]),
       torch.tensor([.08,.01])]

'online optimization'
#w1_tot, w2_tot = [], []
#spk_tot = []
for k in range(len(w_0)):
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    neuron.w = nn.Parameter(w_0[k].clone()).to(par.device)
    'optimization'
    E, w1, w2, v, spk = train(par,neuron,x_data,
                                          online=True)
    
    w1_tot.append(w1)
    w2_tot.append(w2)
    spk_tot.append(spk)

#%%

fig = plt.figure(figsize=(5,5), dpi=300)
for k in range(len(w1_tot)):
    plt.plot(w2_tot[k],w1_tot[k],linewidth=2,color='mediumvioletred')
plt.xlim(0,.15)
plt.ylim(0,.15)
    
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$w_2$')
plt.ylabel(r'$w_1$')
plt.savefig(par.dir+'parameter_space.png',format='png', dpi=300)


#%%

c = ['mediumvioletred','firebrick','navy','royalblue']
fig = plt.figure(figsize=(5,4), dpi=300)
plt.plot(w2_tot[0],w1_tot[0],linewidth=2,color=c[0])
plt.plot(w2_tot[4],w1_tot[4],linewidth=2,color=c[1])    
plt.plot(w2_tot[2],w1_tot[2],linewidth=2,color=c[2])  
plt.plot(w2_tot[6],w1_tot[6],linewidth=2,color=c[3])    
plt.xlim(0,.12)
plt.ylim(0,.12)
plt.grid()   
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$w_2$')
plt.ylabel(r'$w_1$')
plt.savefig(savedir+'parameter_space.png',format='png', dpi=300) 

#%%

neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_00 = torch.tensor([.1,.1])
par.epochs = 600
neuron.w = nn.Parameter(w_00.clone()).to(par.device)
'optimization'
E, w1, w2, v, spk = train(par,neuron,x_data,
                                      online=True)