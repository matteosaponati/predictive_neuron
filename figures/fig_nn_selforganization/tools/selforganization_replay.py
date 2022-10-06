"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"selforganization_replay.py"
compute how many neurons are needed for replay

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

from predictive_neuron import models

'---------------------------------------'
import torch.nn.functional as F
def get_subseqs(par,timing):            
    x_data  = []
    
    for n in range(par.nn):

        x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
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

def forward(par,network,x_data):
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    for t in range(par.T):     
        'append voltage state'
        v.append(network.v.clone().detach().numpy())
        'forward pass'
        network(x_data[:,t]) 
        'append output spikes'
        for n in range(par.nn):
            for b in range(par.batch):
                if network.z[b,n] != 0: z[n][b].append(t*par.dt)          
        
    return network, np.stack(v,axis=1), z
'---------------------------------------'

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
par.Dt=2
par.delay = 4
par.jitter_noise = True
par.jitter = 2
par.fr_noise = True
par.freq = .01
par.w_0 = .02
par.T = int((par.n_in*par.delay+(par.n_in*par.Dt)+80)/par.dt)

par.loaddir = ''
par.savedir = '/Users/saponatim/Desktop/'

'---------------------------------------'
'get results'
z_out = np.load(par.loaddir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                    par.n_in,par.nn,par.delay,par.Dt,
                                    par.tau_m,par.v_th,par.w_0),allow_pickle=True).tolist()
w = np.load(par.loaddir+'w_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                    par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th,par.w_0))

'---------------------------------------'


'run across epochs'
n_needed = np.zeros(w.shape[0])
count = 0
for e in range(400):
    print('epoch '+str(e))
    
    'run across neurons in the network'
    for subseq in range(1,par.nn+1):
        print('# neurons: '+str(subseq))
        
        'define sequence recall'
        par.subseq = [x for x in range(subseq)]
        timing = [[] for n in range(par.nn)]
        spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
        for n in range(par.nn): 
            for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)
    
        'create input'
        x_data = get_subseqs(par,timing)
    
        'run dynamics'
        network = models.NetworkClass_SelfOrg(par)
        network.w = nn.Parameter(torch.from_numpy(w[e,:])).to(par.device)
        network.state()
        network, v, z = forward(par,network,x_data)
        
        count = 0
        for n in range(par.nn):
            if z[n][0] != []: count+=1
        
        print('total neurons triggered: '+str(count))
        if count == par.nn:
            n_needed[e] = subseq
            break
        else:

            continue
'---------------------------------------'
'plot'
fig = plt.figure(figsize=(5,6), dpi=300)
plt.plot(n_needed,linewidth=2,color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel(r'# neurons for replay')
plt.ylim(0,9)
plt.savefig(par.savedir+'n_needed.png',format='png', dpi=300)
plt.savefig(par.savedir+'n_needed.pdf',format='pdf', dpi=300)
plt.close('all')             