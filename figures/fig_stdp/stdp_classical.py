import numpy as np
import torch
import torch.nn as nn
import types
import matplotlib.pyplot as plt

from predictive_neuron import models, funs
import tools

savedir = '/gs/home/saponatim/'

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 2e-4
par.tau_m = 15.
par.v_th = 2.

par.tau_x = 2.

par.N = 2
par.N_stdp = 2
par.batch = 1
par.T = int(500/par.dt)
par.epochs = 60

'initial conditions'
w_0_pre = np.array([.001,.11])
w_0_post = np.array([.11,.001])

'optimization'
par.bound='soft'

timing = np.array([2.,6.])/par.dt

x_data,density = funs.get_sequence_stdp(par,timing)
neuron = models.NeuronClass(par)
neuron.w = nn.Parameter(torch.tensor([.001,.11])).to(par.device)

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

w1_pre,w2_pre,spk_prepost = train(par,neuron,x_data)

'initial condition'
w_0_pre = torch.tensor([.001,.08])
w_0_post = torch.tensor([.08,.001])
delay = np.arange(4.,40,10)

w1,w2 = [[],[]],[[],[]]
spk = [[],[]]
for k in range(len(delay)):
    print('delay '+str(delay[k]))
    
    timing = np.array([2.,2.+ delay[k]])/par.dt
    par.T = int(delay[k]/par.dt) + 200
    x_data,density = funs.get_sequence_stdp(par,timing)
    
    neuron = models.NeuronClass(par)
    neuron.w = nn.Parameter(w_0_pre).to(par.device)
    w1_pre,w2_pre,spk_prepost = tools.train(par,neuron,x_data)
    
    neuron = models.NeuronClass(par)
    neuron.w = nn.Parameter(w_0_post).to(par.device)
    w1_post,w2_post,spk_postpre = tools.train(par,neuron,x_data)
    
    w1[0].append(w1_pre)
    w1[1].append(w1_post)
    w2[0].append(w2_pre)
    w2[1].append(w2_post)
    spk[0].append(spk_prepost)
    spk[1].append(spk_postpre)

savedir='/gs/home/saponatim/predictive_neuron/figures/fig_stdp/'

np.save(savedir+'w1',w1)
np.save(savedir+'w2',w2)
np.save(savedir+'spk',spk)

wpre = np.load(savedir+'w_pre.npy')
wpost = np.load(savedir+'w_post.npy')
spk = np.load(savedir+'spk.npy',allow_pickle=True)
