"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_example.py":
    
    - example of neural dynamics and synaptic plasticity 
    
Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

par = types.SimpleNamespace()

'architecture'
par.N = 4
par.T = 1000
par.batch = 1
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 3e-5
par.tau_m = 10.
par.v_th = 2.2
par.tau_x = 2.

'set inputs'
timing = np.array([0.,4.,15.,19.])/par.dt
x_data,density = funs.get_sequence(par,timing)

'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    w = []
    
    for t in range(par.T):    
        v.append(neuron.v)      
        neuron(x_data[:,t])       
        with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online()
        w.append(neuron.w.detach())
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return torch.stack(v,dim=1), z, torch.stack(w,dim=1)
'----------------'

'online optimization'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')
w_0 = .05
neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)

neuron.state()
v, z, w = forward(par,neuron,x_data)

'-----------------'
'plots'

'voltage'
fig = plt.figure(figsize=(7,4), dpi=300)
plt.plot(v[0,:].detach(),linewidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('v_example.svg', format='svg', dpi=300)
plt.savefig('v_example.pdf', format='pdf', dpi=300)
plt.close('all')

'weights'
c=['mediumvioletred','navy','lightseagreen','firebrick']
fig = plt.figure(figsize=(7,4), dpi=300)
for k in range(par.N):
    plt.plot(w[k,:].detach(),linewidth=2,color=c[k])
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w_example.svg', format='svg', dpi=300)
plt.savefig('w_example.pdf', format='pdf', dpi=300)
plt.close('all')
