import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs
import torch.nn.functional as F

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
    v_out, spk_out = [], []
    
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
        spk_out.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return loss_out, w, v_out, spk_out

#%%
'-------------------'    
par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 1.5
par.tau_x = 2.

'architecture'
par.N = 100
par.spk_volley = 'random'
par.Dt = 4.
par.T = int((2*par.N) // par.dt)
par.seed = 1992
par.batch = 1
par.epochs = 1500
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .03
#
#par.init = 'trunc_gauss'
#par.init_mean = 1.
#par.init_a = 0.
#par.init_b = 2.

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'

#%%

timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
x_data, density = funs.get_sequence(par,timing)

"""
IMP:
    - compute density of pattern and density of noise
    - show the effect of the two components on learning the sequence
    - show how this can depend on neuronal parameters
"""

#%%

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')

#%%
loss, w, v, spk = train(par,x_data)

#%%

fig = plt.figure(figsize=(7,11), dpi=300)
plt.subplot(3,1,1)
plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')
plt.xticks(np.arange(par.T)[::500],np.linspace(0,par.T*par.dt,par.T)[::500].astype(int))
#for k in range(len(spk[-1])):
#    plt.axvline(x = spk[-1][k]/par.dt,color='mediumvioletred')
plt.xlabel('time [ms]')
plt.xlim(0,par.T)
plt.ylabel('inputs')
plt.subplot(3,1,2)
plt.plot(np.array(density)/(par.N),linewidth=2,color='navy')
for k in range(len(spk[-1])):
    plt.axvline(x = spk[-1][k],color='mediumvioletred')
plt.xlabel('time [ms]')
plt.xlim(0,int(par.T*par.dt))
plt.ylabel(r'fr [1/$\tau_m$]')
plt.subplot(3,1,3)
plt.pcolormesh(w.T,cmap='coolwarm')
plt.colorbar()
plt.xlabel('epochs')
plt.ylabel(r'$\vec{w}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk_density.png',format='png', dpi=300)
plt.close('all')






##########################################

#%%

'-------------------'    
par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'architecture'
par.spk_volley = 'random'
par.sequences = 5
par.N_sequences = [10*k + np.arange(10,dtype=int) for k in range(par.sequences)]
par.DT = int(10/par.dt)
par.N = np.sum(10*par.sequences,dtype=int)
par.Dt = 4.
par.T = int(((par.Dt*par.N/par.dt)))
par.seed = 1992
par.batch = 1
par.epochs = 1200
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .04

#par.init = 'trunc_gauss'
#par.init_mean = 1.
#par.init_a = 0.
#par.init_b = 2.

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'


#%%

timing = []
for k  in range(par.sequences):
    
    timing.append(k*(par.Dt*len(par.N_sequences[k]))/par.dt + np.cumsum(np.random.randint(0,par.Dt,len(par.N_sequences[k]))/par.dt))


#%%
x_data, density = funs.get_multi_sequence(par,timing)

#%%

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')


#%%
loss, w, v, spk = train(par,x_data)


#%%


fig = plt.figure(figsize=(7,11), dpi=300)
plt.subplot(3,1,1)
plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')
plt.xticks(np.arange(par.T)[::500],np.linspace(0,par.T*par.dt,par.T)[::500].astype(int))
#for k in range(len(spk[-1])):
#    plt.axvline(x = spk[-1][k]/par.dt,color='mediumvioletred')
plt.xlabel('time [ms]')
plt.xlim(0,par.T)
plt.ylabel('inputs')
plt.subplot(3,1,2)
plt.plot(np.array(density)/(par.N),linewidth=2,color='navy')
for k in range(len(spk[-1])):
    plt.axvline(x = spk[-1][k],color='mediumvioletred')
plt.xlabel('time [ms]')
plt.xlim(0,int(par.T*par.dt))
plt.ylabel(r'fr [1/$\tau_m$]')
plt.subplot(3,1,3)
plt.pcolormesh(w.T,cmap='coolwarm')
plt.colorbar()
plt.xlabel('epochs')
plt.ylabel(r'$\vec{w}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk_density.png',format='png', dpi=300)
plt.close('all')


#%%
fig = plt.figure(figsize=(6,5), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk.png',format='png', dpi=300)
plt.close('all')
