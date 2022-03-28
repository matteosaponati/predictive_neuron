import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt

from predictive_neuron import models, funs

par = types.SimpleNamespace()
'architecture'
par.N = 2
par.T = 300
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
timing = np.array([2.,6.])/par.dt
# timing = np.array(np.arange(0,200,2))/par.dt
x_data = funs.get_sequence(par,timing)

'model'
neuron = models.NeuronClass(par)
# neuron.w = nn.Parameter(.1*torch.ones(par.N)).to(par.device)
loss = nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(neuron.parameters(),
                              lr=par.eta,betas=(.9,.999))
# optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
#%%

E_out = []
w1, w2 = [], []
v_out = []
for e in range(par.epochs):
    
    neuron.state()
    neuron, v = models.num_solution(par,neuron,x_data)
    
    x_hat = torch.einsum("bt,j->btj",v,neuron.w)
    E = loss(x_hat,x_data)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    E_out.append(E.item())
    w1.append(neuron.w[0].item())
    w2.append(neuron.w[1].item())
    v_out.append(v)
    
    if e%50 == 0: print('loss {}'.format(E.item()))
    
    
    
    
#%%

e = -1
plt.plot(v_out[e][0].detach().numpy())
    
    
    
