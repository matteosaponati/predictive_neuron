   
import torch
import torch.nn as nn
import numpy as np
import types
import torch.nn.functional as F

from predictive_neuron import models, funs

import matplotlib.pyplot as plt


'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], [[] for k in range(par.batch)]
    
    for t in range(par.T):            

        v.append(neuron.v)              

        neuron(x_data[:,t])     
        for k in range(par.batch):
            if neuron.z[k] != 0: z[k].append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'----------------'
def train(par,x_data):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'set model'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
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
    v_out, spk_out = [], [[] for k in range(par.batch)]
    
    for e in range(par.epochs):
        
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
        for k in range(par.batch):
            spk_out[k].append(z[k])
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return loss_out, w, v_out, spk_out



'-------------------'    
par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = .3
par.tau_x = 2.

'architecture'
par.N = 100
par.N_sub = 2
par.spk_volley = 'random'
par.Dt = 4.
par.T = int((2*par.N_sub+10) // par.dt)
par.seed = 1992
par.batch = int(par.N/par.N_sub)
par.epochs = 400
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .03

# par.init = 'trunc_gauss'
# par.init_mean = .03
# par.init_a = 0.01
# par.init_b = .05

'additional parameters'
par.savedir = '/Users/matteosaponati/Desktop/predictive_neuron'
par.device = "cpu"
par.tau_x = 2.


#%%

def sequence_capacity_random(par):
    
    'create sequence'
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)   
    for k in range(par.batch):
        timing = np.cumsum(np.random.randint(0,par.Dt,par.N_sub))/par.dt
        x_data[k,timing,k*par.N_sub + np.arange(par.N_sub)] = 1

    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)

def sequence_capacity(par,timing):
    
    'create sequence'
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)   
    for k in range(par.batch):
        x_data[k,timing,k*par.N_sub + np.arange(par.N_sub)] = 1

    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)


#%%

x_data = sequence_capacity_random(par)

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')

#%%

loss, w, v, spk = train(par,x_data)

#%%
plt.pcolormesh(w.T,cmap='coolwarm')

#%%
batch = 3

for k,j in zip(spk[batch],range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')

#%%




timing = np.linspace(2,2*par.N_sub,par.N_sub)/par.dt

x_data = sequence_capacity(par,timing)

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')

#%%

loss, w, v, spk = train(par,x_data)

#%%
plt.pcolormesh(w.T,cmap='coolwarm')

#%%
batch = 0

for k,j in zip(spk[batch],range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')

#%%
'create input data'
    
timing = np.linspace(4,4*par.N,par.N)/par.dt
step = int(par.N/par.batch)
x_data = funs.get_sequence_capacity2(par,timing,step)