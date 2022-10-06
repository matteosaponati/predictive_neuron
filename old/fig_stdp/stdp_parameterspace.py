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
par.N = 2
par.N_stdp = 2

par.batch = 1

par.device = 'cpu'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.
par.epochs = 600

'set inputs'

par.w_0 = .01

delay = np.arange(4.,40,10)

w, spk = [], []
for k in range(len(delay)):
    print('delay '+str(delay[k]))
    
    timing = np.array([2.,2.+ delay[k]])/par.dt
    par.T = int(delay[k]/par.dt) + 200
    x_data,density = funs.get_sequence_stdp(par,timing)
    
    w_temp,v,spk_temp = tools.train_parspace(par,x_data)
    w.append(w_temp)
    spk.append(spk_temp)
    
np.save(savedir+'w_parspace',w)
np.save(savedir+'spk_parspace',spk) 
    

#%%
savedir = '/gs/home/saponatim/'
w = np.load(savedir+'w_parspace.npy')
spk = np.load(savedir+'spk_parspace.npy',allow_pickle=True).tolist()

w_fin = np.zeros((2,len(delay)))

for k in range(len(delay)):
    w_fin[:,k] = w[k,-1,:]
    
    
    
    
#%%

#timing = np.array([2.,26.])/par.dt
#par.T = 600
#par.epochs = 600
#x_data,density = funs.get_sequence_stdp(par,timing)
#neuron = models.NeuronClass(par)
#neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
#w1,w2,spk = tools.train(par,neuron,x_data)
#
#savedir='/gs/home/saponatim/predictive_neuron/figures/fig_stdp/'
#
#np.save(savedir+'w1_parspace',w1)
#np.save(savedir+'w2_parspace',w2)
#np.save(savedir+'spk_parspace',spk)

#%%
#
#w1 = np.load(savedir+'w1_parspace.npy')
#w2 = np.load(savedir+'w2_parspace.npy')
#spk = np.load(savedir+'spk_parspace.npy',allow_pickle=True)