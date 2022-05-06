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

'architecture'
par.N = 2
par.T = 200
par.batch = 1
par.epochs = 500
par.device = 'cpu'
par.seed = 1992

'model parameters'
par.dt = 1.
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 10.
par.tau_x = 2.

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data, density = funs.get_sequence(par,timing)

par.init = 'fixed'
par.w_0 = .03

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'

#%%

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')

#%%
loss, w, v, spk = train(par,x_data)


#%%
x_01 = x_data[0][:,0].detach().numpy().copy()
#%%
x_05 = x_data[0][:,0].detach().numpy().copy()
#%%
x_1 = x_data[0][:,0].detach().numpy().copy()

#%%

w_01 = w.copy()

v_01 = v[0][0,:].copy()


#%%

w_05 = w.copy()

v_05 = v[0][0,:].copy()

#%%

w_1 = w.copy()

v_1 = v[0][0,:].copy()


#%%


plt.plot(v_01)
plt.plot(v_05)
plt.plot(v_1)

plt.plot(x_01)
plt.plot(x_05)
plt.plot(x_1)

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
plt.ylim(0,10)
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk.png',format='png', dpi=300)
plt.close('all')




##########################################

#%%
'-------------------'    
par = types.SimpleNamespace()

'model parameters'
par.dt = .5
par.eta = 1e-3
par.tau_m = 10.
par.v_th = .5
par.tau_x = 2.

'architecture'
par.N = 500
par.spk_volley = 'random'
par.Dt = 4.
par.T = int((2*par.N) // par.dt)
par.seed = 1992
par.batch = 1
par.epochs = 1000
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .03

#par.init = 'trunc_gauss'
#par.init_mean = .05
#par.init_a = 0.
#par.init_b = .1

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

#%%
fig = plt.figure(figsize=(6,5), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')
plt.xlim(0,1000)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk.png',format='png', dpi=300)
plt.close('all')




