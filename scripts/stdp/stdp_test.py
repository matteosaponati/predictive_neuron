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
par.T = 500
par.batch = 1
par.epochs = 600
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 2e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data,_ = funs.get_sequence(par,timing)

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
def train(par,x_data,online=False,bound=False):
    
    'allocate outputs'
    E_out = []
    w1, w2 = [], []
    v_out, spk_out = [],[]
    
    neuron = models.NeuronClass(par)
    neuron.w = nn.Parameter(w_0_pre).to(par.device)
    optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
    loss = nn.MSELoss(reduction='sum')
    
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

'----------------'

def stdp_window(delay,par,w_0_pre,w_0_post):
    w_pre,w_post = [],[]
    
    for k in range(len(delay)):
        'set pairing'
        'set inputs'
        timing = np.array([2.,2+delay[k]])/par.dt
        x_data,density = funs.get_sequence(par,timing)
        
        
        'numerical solution'
        E_pre, w1_pre, w2_pre, v_pre, spk_pre = train(par,x_data)
        
        E_post, w1_post, w2_post, v_post, spk_post = train(par,x_data)
        
        'get weights'
        w_pre.append(w1_pre[-1][0,-1])
        w_post.append(w2_pre[-1][1,-1])

    return w_pre, w_post



#%%

delay = np.arange(4.,60,.05)

w_0_pre = torch.Tensor([.001,.11])
w_0_post = torch.Tensor([.11,.001])


w_pre, w_post = stdp_window(delay,par,w_0_pre,w_0_post)



#%%

'online optimization'
neuron = models.NeuronClass(par)
loss = nn.MSELoss(reduction='sum')

w_0_pre = torch.Tensor([.001,.11])

neuron.w = nn.Parameter(w_0_pre).to(par.device)

delay = np.arange(4.,60,.05)