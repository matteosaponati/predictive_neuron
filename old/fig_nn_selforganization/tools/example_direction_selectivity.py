"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"example_direction_selectivity.py"
neural network with self-organization lateral connections - network effect 
for direction selectivity

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
'preferred direction'
c = ['paleturquoise','lightseagreen','lightblue','dodgerblue','royalblue','mediumblue','mediumslateblue','midnightblue']

timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)

x_data = funs.get_sequence_nn_selforg_noise(par,timing)
network = models_nn.NetworkClass_SelfOrg(par)
network.w = nn.Parameter(torch.from_numpy(w[-1,:])).to(par.device)
network.state()
network, v, z = network(par,network,x_data)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(z[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_preferred_dir.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_preferred_dir.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v[0,:,:].copy()
v_spike[v_spike>3.5]=9
fig = plt.figure(figsize=(5,4), dpi=300)
count = 0
for n in range(0,par.nn,1):
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=c[count])
    count +=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.savefig(par.savedir+'v_preferred_dir.png',format='png', dpi=300)
plt.savefig(par.savedir+'v_preferred_dir.pdf',format='pdf', dpi=300)
plt.close('all') 

'---------------------------------------'
'null direction'

timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)[::-1]/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)

x_data = funs.get_sequence_nn_selforg_noise(par,timing)
network = models_nn.NetworkClass_SelfOrg(par)
network.w = nn.Parameter(torch.from_numpy(w[-1,:])).to(par.device)
network.state()
network, v, z = network(par,network,x_data)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(z[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')
plt.ylabel(r'$N_{nn}$')
plt.savefig(par.savedir+'spk_null_dir.png',format='png', dpi=300)
plt.savefig(par.savedir+'spk_null_dir.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v[0,:,:].copy()
v_spike[v_spike>3.5]=9
fig = plt.figure(figsize=(5,4), dpi=300)
count = 0
for n in range(0,par.nn,1):
    plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=c[count])
    count +=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.savefig(par.savedir+'v_null_dir.png',format='png', dpi=300)
plt.savefig(par.savedir+'v_null_dir.pdf',format='pdf', dpi=300)
plt.close('all') 
