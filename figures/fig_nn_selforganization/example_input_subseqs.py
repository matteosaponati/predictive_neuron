"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"example_input_subseqs.py"
neural network with self-organization lateral connections for 
sequence recall and sequence completion

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn as nn
import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models_nn, funs

'---------------------------------------'
'define function for input subseqs'
import torch.nn.functional as F
def get_subseqs(par,timing):            
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.fr_noise == True:
            prob = par.freq*par.dt
            mask = torch.rand(par.batch,par.T,par.n_in).to(par.device)
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            x[mask<prob] = 1        
        else:
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            
        'create sequence + jitter' 
        if n in par.subseq:
         x[b,timing[n][b],range(par.n_in)] = 1
        
        'filtering'
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
            
        'add to total input'
        x_data.append(x.permute(0,2,1))

    return torch.stack(x_data,dim=3)

'---------------------------------------'
'create parameter structure'
par = types.SimpleNamespace()
'architecture'
par.n_in = 26
par.nn = 8
par.batch = 1
par.lateral = 2
par.device = 'cpu'
par.dtype = torch.float
'model parameters'
par.dt = .05
par.tau_m = 15.
par.v_th = 3.5
par.tau_x = 2.
par.is_rec = True
par.online = True
'input'
par.delay = 4
par.jitter_noise = True
par.jitter = 2
par.fr_noise = True
par.freq = .01
par.w_0 = .02
par.T = int((par.n_in*par.delay+(par.n_in*par.Dt)+80)/par.dt)

par.loaddir = ''
par.savedir = ''

'---------------------------------------'
'get results'
z_out = np.load(par.loaddir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                    par.n_in,par.nn,par.delay,par.Dt,
                                    par.tau_m,par.v_th,par.w_0),allow_pickle=True).tolist()
w = np.load(par.loaddir+'w_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                    par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th,par.w_0))

'---------------------------------------'
'define sequence recall'
par.subseq = [0,1]
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)

'---------------------------------------'
'create network dynamics - first epoch'
x_data = funs.get_subseqs(par,timing)
network = models_nn.NetworkClass_SelfOrg(par)
network.w = nn.Parameter(torch.from_numpy(w[0,:])).to(par.device)
network.state()
network, v_before, z_before = network(par,network,x_data)

'---------------------------------------'
'create network dynamics - final epoch'
x_data = funs.get_subseqs(par,timing)
network = models_nn.NetworkClass_SelfOrg(par)
network.w = nn.Parameter(torch.from_numpy(w[-1,:])).to(par.device)
network.state()
network, v_after, z_after = network(par,network,x_data)

'---------------------------------------'
'plot network activity'
c = ['paleturquoise','lightseagreen','lightblue','dodgerblue','royalblue','mediumblue','mediumslateblue','midnightblue']

'before'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(z_before[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_before_recall.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_before_recall.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v_before[0,:,:].copy()
v_spike[v_spike>3.5]=9
fig = plt.figure(figsize=(5,4), dpi=300)
count = 0
for n in range(0,par.nn,1):
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=c[count])
    count +=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.savefig(par.savedir+'v_before_recall.png',format='png', dpi=300)
plt.savefig(par.savedir+'v_before_recall.pdf',format='pdf', dpi=300)
plt.close('all') 

'after'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(z_after[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_after_recall.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_after_recall.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v_after[0,:,:].copy()
v_spike[v_spike>3.5]=9
fig = plt.figure(figsize=(5,4), dpi=300)
count = 0
for n in range(0,par.nn,1):
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=c[count])
    count +=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.savefig(par.savedir+'v_after_recall.png',format='png', dpi=300)
plt.savefig(par.savedir+'v_after_recall.pdf',format='pdf', dpi=300)
plt.close('all') 

'---------------------------------------'
'define sequence completion'
par.subseq = [0,1,2,5,6,7]
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)

'---------------------------------------'
'create network dynamics - first epoch'
x_data = funs.get_subseqs(par,timing)
network = models_nn.NetworkClass_SelfOrg(par)
network.w = nn.Parameter(torch.from_numpy(w[0,:])).to(par.device)
network.state()
network, v_before, z_before = network(par,network,x_data)

'---------------------------------------'
'create network dynamics - final epoch'
x_data = funs.get_subseqs(par,timing)
network = models_nn.NetworkClass_SelfOrg(par)
network.w = nn.Parameter(torch.from_numpy(w[-1,:])).to(par.device)
network.state()
network, v_after, z_after = network(par,network,x_data)

'---------------------------------------'
'plot network activity'
c = ['paleturquoise','lightseagreen','lightblue','dodgerblue','royalblue','mediumblue','mediumslateblue','midnightblue']

'before'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(z_before[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_before_completion.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_before_completion.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v_before[0,:,:].copy()
v_spike[v_spike>3.5]=9
fig = plt.figure(figsize=(5,4), dpi=300)
count = 0
for n in range(0,par.nn,1):
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=c[count])
    count +=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.savefig(par.savedir+'v_before_completion.png',format='png', dpi=300)
plt.savefig(par.savedir+'v_before_completion.pdf',format='pdf', dpi=300)
plt.close('all') 

'after'
fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(z_after[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_after_completion.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_after_completion.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v_after[0,:,:].copy()
v_spike[v_spike>3.5]=9
fig = plt.figure(figsize=(5,4), dpi=300)
count = 0
for n in range(0,par.nn,1):
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=c[count])
    count +=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.savefig(par.savedir+'v_after_completion.png',format='png', dpi=300)
plt.savefig(par.savedir+'v_after_completion.pdf',format='pdf', dpi=300)
plt.close('all') 
