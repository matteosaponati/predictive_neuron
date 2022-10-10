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
import torch

'--------------'

def forward_NumPy(par,neuron,x_data):    
    v,z = [], []
    loss = []
    for t in range(par.T):    
        v.append(neuron.v)
        loss.append(np.linalg.norm(x_data[:,t] - neuron.v*neuron.w))
        neuron(x_data[:,t])          
        if neuron.z != 0: z.append(t*par.dt)    
    return neuron, v, z, loss

'--------------'

'numerical solution and training for sequences - NumPy version'
def train_NumPy(par,neuron,x_data):
    w1, w2 = [], []
    v_tot, spk_tot = [],[]
    loss_tot = []
    for e in range(par.epochs):        
        neuron.state()
        neuron, v, spk, loss = forward_NumPy(par,neuron,x_data)    
        v_tot.append(v)
        spk_tot.append(spk)
        loss_tot.append(np.sum(loss))
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%100 == 0: print(e)        
    return w1, w2, v_tot, spk_tot, loss_tot

'--------------'

'numerical solution and training for STDP - NumPy version'

def train_STDP(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, v, z, loss = forward_NumPy(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print(e)        
    return w1, w2
