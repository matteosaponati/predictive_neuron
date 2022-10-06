"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig2_capacity.py":
    
    - numerical investigation of the capacity of the model
    - dependence on overlap between synapses
    - dependence on model parameters

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
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

savedir = '/gs/home/saponatim/'

'set model'
par = types.SimpleNamespace()
par.epochs = 1000
par.device = 'cpu'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.

'set inputs'
par.Dt = 4
par.batch = 4
par.N_sub = 10
par.N = int(par.batch*par.N_sub)
par.T = int((par.N_sub*par.Dt)/par.dt)

'noise'
par.offset = 'False'
par.fr_noise = 'False'
par.freq = .002
par.jitter_noise = 'False'
par.jitter = 2
par.seed = 1992
par.init = 'fixed'
par.w_0 = .03

#%%
'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(neuron.v)              

        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'----------------'
def train(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'set model' 
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
    x_data = funs.sequence_capacity(par,timing)
    
    'initialization'
    if par.init == 'trunc_gauss':
        neuron.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(neuron.w, mean=par.init_mean, std=.1/np.sqrt(par.N),
                                    a=par.init_a,b=par.init_b)
    if par.init == 'fixed':
        neuron.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
        
    'optimizer'
    optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    
    'allocate outputs'
    loss_out = []
    w = np.zeros((par.epochs,par.N))
    v_out, spk_out = [], []
    
    for e in range(par.epochs):
        
#        x_data = funs.sequence_capacity(par,timing)
        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)
        
        'evaluate loss'
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x_data)
        optimizer.zero_grad()
        E.backward()
        optimizer.step()
        
        'output'
        loss_out.append(E.item())
        w[e,:] = neuron.w.detach().numpy()
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return loss_out, w, v_out, spk_out

#%%
loss, w, v, spk = train(par)

#%%
' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_capacity.png',format='png', dpi=300)
plt.savefig('w_capacity.pdf',format='pdf', dpi=300)
plt.close('all')

#%%

'examples'

delay = 40

timing = []
for k in range(par.batch):
    timing.append((k*delay + np.cumsum(np.random.randint(0,par.Dt,par.N_sub)))/par.dt)
timing = np.array(timing).flatten()
par.T = int((par.batch*(par.N_sub*par.Dt + delay)/2)/par.dt)
par.fr_noise = 'True'
par.freq = .005
x_data, density, fr = funs.get_sequence(par,timing)

plt.pcolormesh(x_data[3,:,:].T,cmap='Greys')

#%%
'end of training'

neuron = models.NeuronClass(par)
neuron.w = nn.Parameter(torch.tensor(w[-1,:],dtype=torch.float32)).to(par.device)
neuron.state()
v = []
z = [[] for b in range(par.batch)]
for t in range(par.T):            
    v.append(neuron.v)
    neuron(x_data[:,t])    
    for b in range(par.batch):
        if neuron.z[b] != 0: z[b].append(t*par.dt)    
    
v = torch.stack(v,dim=1)

v_spike = v[2,:].detach().clone().numpy()
v_spike[v_spike>2.5]=7

model_example(par,savedir,z[2],v_spike)

neuron = models.NeuronClass(par)
neuron.w = nn.Parameter(torch.tensor(w[0,:],dtype=torch.float32)).to(par.device)
neuron.state()
v = []
z = [[] for b in range(par.batch)]
for t in range(par.T):            
    v.append(neuron.v)              
    neuron(x_data[:,t])        
    for b in range(par.batch):
        if neuron.z[b] != 0: z[b].append(t*par.dt)    
    
v = torch.stack(v,dim=1)
plt.plot(v[1,:].detach().numpy())

v_spike = v[2,:].detach().clone().numpy()
v_spike[v_spike>2.5]=7

model_example(par,savedir,z[2],v_spike)

#%%

def model_example(par,savedir,spk,v):
    
    '1. capacity example'
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.eventplot(z,linelengths = 3,linewidths = .5,colors = 'purple')
    plt.xlim(0,par.T*par.dt)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'spk_epoch.png',format='png', dpi=300)
    plt.savefig(savedir+'spk_epoch.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v,linewidth=2,color='navy')
    plt.xlim(0,par.T*par.dt)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('v')
    plt.savefig(savedir+'v_epoch_0.png',format='png', dpi=300)
    plt.savefig(savedir+'v_epoch_0.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    return







#%%


'4. weights dynamics'
hex_list = ['#FFFAF0','#7D26CD']
#divnorm = colors.DivergingNorm(vmin=w.min(),vcenter=0, vmax=w.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list),aspect='auto')
plt.colorbar()

#%%

for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)
plt.ylim(0,60)

#%%


def capacity_example(par,savedir,delay=40):    
    timing = []
    for k in range(par.batch):
        timing.append((k*delay + np.cumsum(np.random.randint(0,par.Dt,par.N_sub)))/par.dt)
    timing = np.array(timing).flatten()
    timing = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N))/par.dt
    par.T = int((par.batch*(par.N_sub*par.Dt + delay)/2)/par.dt)

    fig = plt.figure(figsize=(4,4), dpi=300)
    offset = 1
    for k in range(par.N):
        bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
        plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
        plt.eventplot([timing[k]],lineoffsets = offset,linelengths = 3,linewidths = .5,colors = 'purple')
        offset += 1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'spk_capacity.png',format='png', dpi=300)
    plt.savefig(savedir+'spk_capacity.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
def capacity_density(par,density):
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.plot(density,linewidth=2,color='purple')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel(r'density [spk/$\tau_m$]')
    plt.savefig(savedir+'capacity_density.png',format='png', dpi=300)
    plt.savefig(savedir+'capacity_density.pdf',format='pdf', dpi=300)
    plt.close('all')
    
    
def model_example(par,savedir,spk,v):
    
    '1. capacity example'
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.eventplot(z[0],linelengths = 3,linewidths = .5,colors = 'purple')
    plt.xlim(0,par.T*par.dt)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'spk_epoch.png',format='png', dpi=300)
    plt.savefig(savedir+'spk_epoch.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.plot(np.arange(0,par.T,par.dt),v,linewidth=2)
    plt.xlim(0,par.T*par.dt)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'v_epoch_0.png',format='png', dpi=300)
    plt.savefig(savedir+'v_epoch_0.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    return
    
    
    
#%%
    
capacity_example(par,savedir)

#%%

delay = 40

timing = []
for k in range(par.batch):
    timing.append((k*delay + np.cumsum(np.random.randint(0,par.Dt,par.N_sub)))/par.dt)
timing = np.array(timing).flatten()
par.T = int((par.batch*(par.N_sub*par.Dt + delay)/2)/par.dt)

x_data, density, fr = funs.get_sequence(par,timing)

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')
#capacity_density(par,density)

#%%


#%%

offset = 1
for k in range(par.N):
    bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
    plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
    plt.eventplot(z[],linelengths = 3,linewidths = .5,colors = 'purple')
    offset += 1


