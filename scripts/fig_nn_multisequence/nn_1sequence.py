"""
n_in = 2,3: T = 500, epochs = 400
n_in = 10: T = 1000, epochs = 800

"""
import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models_nn as models
from predictive_neuron import funs

'----------------'
def forward(par,network,x_data,online=False):
    
    v,z = [], [[] for n in range(par.nn)]
    
    for t in range(par.T):            
        
        v.append(network.v.clone())    

        if online: 
            with torch.no_grad():
                network.backward_online(x_data[:,t])
                network.update_online()              
        network(x_data[:,t]) 
        
        for n in range(par.nn):
            if network.z[0,n] != 0: z[n].append(t*par.dt)  
        
    return network, torch.stack(v,dim=1), z
'----------------'

#%%
par = types.SimpleNamespace()

'architecture'
par.n_in = 3
par.nn = 2
par.T = 500
par.batch = 1
par.epochs = 400
par.device = 'cpu'
par.dtype = torch.float

'model parameters'
par.dt = .05
par.eta = 1e-5
par.tau_m = 10
par.v_th = 2.5
par.tau_x = 2.

par.is_rec = True

'set inputs'
par.Dt = 4
timing = [[] for n in range(par.nn)]
for n in range(par.nn):
    for b in range(par.batch):
        timing[n].append(np.arange(0,par.Dt*par.n_in,par.Dt)/par.dt)
        
x_data = funs.get_sequence_nn(par,timing)

'set model'
network = models.NetworkClass(par)
par.w_0 = .1
par.w_0rec = -.05
network.w = nn.Parameter(torch.FloatTensor(par.n_in,par.nn).uniform_(0.,par.w_0))
if par.is_rec == True: 
    w_rec=  par.w_0rec*np.ones((par.nn,par.nn))
    w_rec = np.where(np.eye(par.nn)>0,np.zeros_like(w_rec),w_rec)
    network.wrec = torch.as_tensor(w_rec,dtype=par.dtype).to(par.device)

#%%
'setup optimization'
loss_fn = nn.MSELoss(reduction='sum')

w = np.zeros((par.epochs,par.n_in,par.nn))
wrec = np.zeros((par.epochs,par.nn,par.nn))
E = [[] for n in range(par.nn)]
z_out = [[] for n in range(par.nn)]
v_out = []

for e in range(par.epochs):
        
    network.state()
    network, v, z = forward(par,network,x_data,online=True)
    
    x_hat = torch.einsum("btn,jn->btjn",v,network.w)
    lossList = []
    for n in range(par.nn):  
        loss = loss_fn(x_hat[:,:,:,n],x_data[:,:,:,n])
        lossList.append(loss)
        
    w[e,:,:] = network.w.detach().numpy()
    if par.is_rec == True: wrec[e,:,:] = network.wrec.detach().numpy()
    for n in range(par.nn):
        z_out[n].append(z[n])
    v_out.append(v)
    
    if e%50 == 0:
        for n in range(par.nn): print('loss {}: {}'.format(n,lossList[n].item()))
        
#%%

'plots'
plt.subplot(2,1,1)
plt.plot(w[:,:,0],linewidth=2)
plt.ylabel(r'$\vec{w}$')
plt.subplot(2,1,2)
plt.plot(w[:,:,1],linewidth=2)
plt.xlabel('epochs')
plt.ylabel(r'$\vec{w}$')

#%%
plt.plot(np.arange(0,par.T*par.dt,par.dt),v_out[-1][0,:,0].detach().numpy(),linewidth=2,color='purple')
plt.plot(np.arange(0,par.T*par.dt,par.dt),v_out[-1][0,:,1].detach().numpy(),linewidth=2,color='navy')
plt.axhline(y=par.v_th,color='k')
plt.xlabel('time [ms]')
plt.ylabel('v')        

#%%

for k in range(len(timing)):
    plt.axhline(y=timing[0][0][k]*par.dt,color='k',linewidth=2,linestyle='dashed')
for k,j in zip(z_out[0],range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)    
for k,j in zip(z_out[1],range(par.epochs)):
    plt.scatter([j]*len(k),k,c='navy',s=7) 