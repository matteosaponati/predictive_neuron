"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"tools.py":
auxiliary functions for synaptic plasticity models
    
Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch.nn as nn
import torch

from predictive_neuron import models

'----------------'
def forward(par,neuron,x_data):
    v,z = [], []
    for t in range(par.T):    
        v.append(neuron.v) 
        with torch.no_grad():
            neuron.backward_online(x_data[:,t])
            neuron.update_online()  
        neuron(x_data[:,t])  
        if neuron.z[0] != 0: z.append(t*par.dt)    
    return neuron, torch.stack(v,dim=1), z

def train(par,neuron,x_data):
    w1, w2 = [], []
    spk_out = []
    'training'
    for e in range(par.epochs):
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)
        'output'
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        spk_out.append(z)
    return w1, w2, spk_out
'---------------------------------------------'

'----------------'
def forward_parspace(par,neuron,x_data):
    v,z = [], []
    for t in range(par.T):
        v.append(neuron.v)              
        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
    return neuron, torch.stack(v,dim=1), z
'----------------'

'----------------'
def train_parspace(par,x_data):

    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    neuron.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
    optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))

    w = np.zeros((par.epochs,par.N))
    v_out, spk_out = [], []
    
    for e in range(par.epochs):
        neuron.state()
        neuron, v, z = forward_parspace(par,neuron,x_data)
        'evaluate loss'
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x_data)
        optimizer.zero_grad()
        E.backward()
        optimizer.step()
        
        w[e,:] = neuron.w.detach().numpy()
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return w, v_out, spk_out