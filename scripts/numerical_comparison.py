import numpy as np
import torch
import types
import torch.nn as nn

from predictive_neuron import models, funs

par = types.SimpleNamespace()
'architecture'
par.N = 2
par.T = 600
par.batch = 1
par.epochs = 800
par.device = 'cpu'
'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 1.5
par.tau_x = 2.
par.freq = 0
'set inputs'
timing = np.array([2.,4.])/par.dt
x_data = funs.get_sequence(par,timing)

'model'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')

'numerical simulation'
grad_online = torch.zeros(1,2)
v = []

neuron.state()
for t in range(par.T):
    
    v.append(neuron.v)
        
    'online evaluation of gradient'
    with torch.no_grad():
        neuron.backward_online(x_data[:,t])
        grad_online += neuron.grad
    
    neuron(x_data[:,t])
    
# 'offline evaluation'
# with torch.no_grad():
#     grad_offline = neuron.backward_offline(torch.stack(v,dim=1),x_data)

'evaluation of gradient via bptt'
x_hat = torch.einsum("bt,j->btj",torch.stack(v,dim=1),neuron.w)
E = .5*loss(x_hat,x_data)
grad_bptt = torch.autograd.grad(E,neuron.w,retain_graph=True)[0]

'numerical comparison'
max_dgrad = torch.max((grad_bptt - grad_online)/grad_bptt)
