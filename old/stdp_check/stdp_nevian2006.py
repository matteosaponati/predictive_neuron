import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

savedir = '/gs/home/saponatim/'

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.T = 300
par.batch = 1
par.epochs = 60
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 1.8e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.


#%%
'----------------'
def forward(par,neuron,x_data,online=False,bound=False):
    
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
def train(par,neuron,x_data,online=False,bound=False):
    
    'allocate outputs'
    E_out = []
    w1, w2 = [], []
    v_out, spk_out = [],[]
    
    loss = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
    
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data,online,bound)
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
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return E_out, w1, w2, v_out, spk_out

#%%

n_spikes = 3
dt_burst, dt = np.array([10,20,50])/par.dt, 10/par.dt
E_pre, E_post = [], []
w1_pre,w2_pre = [],[]
w1_post,w2_post = [],[]
v_pre, spk_pre = [],[]
v_post, spk_post = [],[]


#%%

for j in dt_burst:
    
    'set inputs'
    print('solving {} dt'.format(j))
    
    par.T = int(j*n_spikes + 100)
    timing = [np.array([0.])/par.dt,np.arange(dt,j*n_spikes + j,j)]
    x_data,_ = funs.get_sequence_stdp(par,timing)
    
    w_0_pre = torch.Tensor([.01,.08])
    neuron = models.NeuronClass(par)
    neuron.w = nn.Parameter(w_0_pre.clone()).to(par.device)
    E, w1, w2, v, spk = train(par,neuron,x_data)
    
    E_pre.append(E)
    w1_pre.append(w1)
    w2_pre.append(w2)
    v_pre.append(v)
    spk_pre.append(spk)
    
    w_0_pre = torch.Tensor([.08,.01])
    neuron = models.NeuronClass(par)
    neuron.w = nn.Parameter(w_0_pre.clone()).to(par.device)
    E, w1, w2, v, spk = train(par,neuron,x_data)
    
    E_post.append(E)
    w1_post.append(w1)
    w2_post.append(w2)
    v_post.append(v)
    spk_post.append(spk)
    
    
