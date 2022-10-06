import numpy as np
import torch
import torch.nn as nn
import types
import matplotlib.pyplot as plt

from predictive_neuron import models, funs
import tools

savedir = '/gs/home/saponatim/'

par = types.SimpleNamespace()
'set model'
par.device = 'cpu'
par.dt = .05
par.eta = 2e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'set inputs'
par.N = 2
par.N_stdp = 2
par.batch = 1
par.T = int(100/par.dt)
x_data,density = funs.get_sequence_stdp(par,timing)

'model initialization'
neuron = models.NeuronClass(par)
neuron.w = nn.Parameter(torch.tensor([.001,.11])).to(par.device)

'optimization'
par.bound='hard'

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
        
        if e%2 == 0: print(e)
    return w1, w2, spk_out
'---------------------------------------------'


'initial conditions'
w_sweep = np.arange(.01,.1,.05)

'parameter space analysis'
def parameter_space(inputs,w_sweep,A_x,tau_x,T,epochs,p_num):
    'initiate matrices'
    w_1,w_2 = np.zeros((len(w_sweep),len(w_sweep))), np.zeros((len(w_sweep),len(w_sweep)))
    'fill matrices'
    for k in range(len(w_sweep)):
        print(k)
        for j in range(len(w_sweep)):
            var = model.train(inputs,inputs.shape[0],epochs,T,np.array([w_sweep[k],w_sweep[j]]),p_num)
            w_1[k,j] = var['w'][-1][0,-1]
            w_2[k,j] = var['w'][-1][1,-1]            
    return w_1 - np.tile(w_sweep,(len(w_sweep),1)).T, w_2 - np.tile(w_sweep,(len(w_sweep),1))