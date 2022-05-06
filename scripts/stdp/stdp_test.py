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
par.epochs = 60
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 5e-6
par.tau_m = 20.
par.v_th = 2.
par.tau_x = 2.

'----------------'

#delay = np.arange(0.,20,1.)

#w_pre, w_post, v_pre, spk_pre, v_post, spk_post  = stdp_window(delay,par)


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
            neuron.w.grad = torch.where(torch.tensor([True,False]),
                                        torch.zeros_like(neuron.w),
                                        neuron.w.grad)
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

def stdp_window(delay,par):
    
    w_pre,w_post = [],[]
    v_pre,spk_pre = [],[]
    v_post,spk_post = [],[]
    E_pre, E_post = [], []
    
    par.T = 200+int(delay[-1]/par.dt)
    for k in range(len(delay)):
        
        'set inputs'
        timing = np.array([0.,delay[k]])/par.dt
        x_data,density = funs.get_sequence(par,timing)
        
        neuron = models.NeuronClass(par)
        w_0_pre = torch.Tensor([.01,.11])
        neuron.w = nn.Parameter(w_0_pre.clone()).to(par.device)
        E, w1_pre, w2_pre, v, spk = train(par,neuron,x_data)
        
        v_pre.append(v)
        spk_pre.append(spk)
        E_pre.append(E)
        
        neuron = models.NeuronClass(par)
        w_0_post = torch.Tensor([.1,.01])
        neuron.w = nn.Parameter(w_0_post.clone()).to(par.device)
        E, w1_post, w2_post, v, spk = train(par,neuron,x_data)
        
        v_post.append(v)
        spk_post.append(spk)
        E_post.append(E)
        
        'get weights'
        w_pre.append(w1_pre[-1])
        w_post.append(w2_post[-1])

    return w_pre, w_post, v_pre, spk_pre, v_post, spk_post

#%%
    
'set inputs'
timing = np.array([2.,20.])/par.dt
x_data,_ = funs.get_sequence(par,timing)

w_0_pre = torch.Tensor([.005,.1])
neuron = models.NeuronClass(par)
neuron.w = nn.Parameter(w_0_pre.clone()).to(par.device)
E_pre, w1_pre, w2_pre, v_pre, spk_pre = train(par,neuron,x_data)

w_0_post = torch.Tensor([.08,.01])
neuron = models.NeuronClass(par)
neuron.w = nn.Parameter(w_0_post.clone()).to(par.device)
E_post, w1_post, w2_post, v_post, spk_post = train(par,neuron,x_data)