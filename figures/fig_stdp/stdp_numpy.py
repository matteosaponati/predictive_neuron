import numpy as np
import types
import matplotlib.pyplot as plt

from predictive_neuron import models, funs
savedir = '/gs/home/saponatim/'

'set model'
par = types.SimpleNamespace()
par.dt = .05
par.eta = 2e-4
par.tau_m = 10.
par.v_th = 2.

par.tau_x = 2.

par.N = 2
par.T = int(500/par.dt)
par.epochs = 10

'initial conditions'
w_0_pre = np.array([.001,.11])
w_0_post = np.array([.11,.001])

timing = np.array([2.,6.])/par.dt
x_data = funs.get_sequence_NumPy(par,timing)

'set model'
par.bound='no'
neuron = models.NeuronClass_NumPy(par)
neuron.w = w_0_pre.copy()

'----------------'
def forward(par,neuron,x_data):
    v,z = [], []
    w = np.zeros((par.N,par.T))
    for t in range(par.T):    
        v.append(neuron.v) 
        neuron(x_data[:,t])  
        if neuron.z != 0: z.append(t*par.dt)
        w[:,t] = neuron.w.copy()
    return neuron, v, z, w

def train(par,neuron,x_data):
    w_tot = []
    spk_out = []
    'training'
    for e in range(par.epochs):
        neuron.state()
        neuron, v, z, w = forward(par,neuron,x_data)
        'output'
        w_tot.append(w)
        spk_out.append(z)
        
        if e%2 == 0: print(e)
    return w_tot, spk_out
'----------------'

#%%

par.bound='soft'
neuron = models.NeuronClass_NumPy(par)
neuron.w = w_0_pre.copy()

neuron.state()
v,z = [], []
w = np.zeros((par.N,par.T))
for t in range(par.T):    
    v.append(neuron.v) 
    neuron(x_data[:,t])  
    print(neuron.w)
    if neuron.z != 0: z.append(t*par.dt)
    w[:,t] = neuron.w.copy()




#%%

w, spk = train(par,neuron,x_data)
