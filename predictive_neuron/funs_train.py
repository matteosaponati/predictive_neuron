"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"funs_train.py"
auxiliary functions to train the single neuron and the network model

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from predictive_neuron import funs

'--------------'
'numerical solution and training for sequences - NumPy version'

def initialize_weights(par,neuron):
    if par.init == 'fixed': 
        return par.init_mean*np.ones(par.N)
    elif par.init == 'random':
        return stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.N)), 
                              (par.init_b-par.init_mean)/(1/np.sqrt(par.N)), 
                              loc=par.init_mean, scale=1/np.sqrt(par.N)).rvs(par.N)
    else: return neuron.w

def forward_NumPy(par,neuron,x_data):    
    v,z = [], []
    loss = []
    for t in range(par.T):    
        v.append(neuron.v)
        loss.append(np.linalg.norm(x_data[:,t] - neuron.v*neuron.w))
        neuron(x_data[:,t])          
        if neuron.z != 0: z.append(t*par.dt)    
    return neuron, v, z, loss

def train_NumPy(par,neuron,x_data=None,timing=None):
#    w1, w2 = [], []
    
    w = []
    v_tot, spk_tot = [],[]
    loss_tot = []
    
    for e in range(par.epochs):     
        
        if x_data == None:
            x_data = funs.get_sequence_NumPy(par,timing)
            
        neuron.state()
        neuron, v, spk, loss = forward_NumPy(par,neuron,x_data)    
        
        v_tot.append(v)
        spk_tot.append(spk)
        loss_tot.append(np.sum(loss))
        w.append(neuron.w.item())
        
#        w1.append(neuron.w[0].item())
#        w2.append(neuron.w[1].item())
        
        if e%100 == 0: print(e)      
        
    return w, v_tot, spk_tot, loss_tot

## might move this into each single script for STDP
def train_STDP(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, v, z, loss = forward_NumPy(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print(e)        
    return w1, w2


'----------------'
def forward_nn_selforg(par,network,x_data):
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    for t in range(par.T):     
        'append voltage state'
        v.append(network.v.clone().detach().numpy())
        'update weights online'
        if par.online == True: 
            with torch.no_grad():
                network.backward_online(x_data[:,t])
                network.update_online()  
        'forward pass'
        network(x_data[:,t]) 
        'append output spikes'
        for n in range(par.nn):
            for b in range(par.batch):
                if network.z[b,n] != 0: z[n][b].append(t*par.dt)          
        
    return network, np.stack(v,axis=1), z
'----------------'

def train_nn_selforg(par):
    
    'create input data'
    x_data = funs.get_sequence_nn_selforg(par)
    
    'set model'
    network = models.NetworkClass_SelfOrg(par)
    
    'initialization'
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.n_in+par.lateral,par.nn)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=.1/np.sqrt(par.par.n_in+par.lateral),
                                    a=par.init_a,b=par.init_b) 
        network.w[par.n_in:,] = par.w_0rec    
    if par.init == 'fixed':
        w = par.w_0*torch.ones(par.n_in+par.lateral,par.nn)
        w[par.n_in:,] = par.w_0rec
        network.w = nn.Parameter(w).to(par.device)
    
    'allocate outputs'
    w = np.zeros((par.epochs,par.n_in+par.lateral,par.nn))
    z_out = [[] for n in range(par.nn)]
    v_out = []
    
    'training'
    for e in range(par.epochs):
        if e%50 == 0: print(e)  
            
        network.state()
        network, v, z = forward(par,network,x_data)
        v_out.append(v)
        
        w[e,:,:] = network.w.detach().numpy()
        for n in range(par.nn):
            z_out[n].append(z[n])

    return w, z_out, v_out
