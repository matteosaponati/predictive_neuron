import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt

from predictive_neuron import models, funs

'----------------'
def num_solution(par,neuron,x_data,online=False,bound=False):
    
    v,z = [], []
    
    for t in range(par.T):    
        v.append(neuron.v)      
        'online update'
        if online: 
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online(bound)    
        'update state variables'        
        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z

'----------------'
def train(par,online=False,bound=False):
    
    'set inputs'
    timing = np.array(par.timing)/par.dt
    x_data = funs.get_sequence(par,timing)
    
    'offline optimization: BPTT with SGD'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')    
    neuron.w = nn.Parameter(torch.abs(neuron.w))
    optimizer = torch.optim.Adam(neuron.parameters(),
                              lr=1e-3,betas=(.9,.999))
    
    'allocate outputs'
    E_out = []
    w1, w2 = [], []
    v_out, spk_out = [],[]
    
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = num_solution(par,neuron,x_data,online,bound)
        'evaluate loss'
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x_data)
        if online == False:
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
        'output'
        E_out.append(E.item())
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        v_out.append(v)
        spk_out.append(z)
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return E_out, w1, w2, v_out, spk_out

'----------------'