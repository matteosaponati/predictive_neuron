"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig_timescales_separation.py":
    
    - numerical difference in synaptic weights depending on the
    voltage and learning timescale

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

savedir = '/gs/home/saponatim/'

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.T = 500
par.batch = 1
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 1e-5
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.
par.freq = 0

'noise'
par.offset = 'False'
par.fr_noise = 'False'
par.jitter_noise = 'False'

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data,densitt,fr = funs.get_sequence(par,timing)

sweep = 500
tau = np.linspace(1,500,sweep)
eta = np.logspace(-8,-1,sweep)

w1_on, w2_on = np.zeros((sweep,sweep)), np.zeros((sweep,sweep))
w1_sgd, w2_sgd = np.zeros((sweep,sweep)), np.zeros((sweep,sweep))

for k in range(sweep):
    
    par.eta = eta[k]
    
    for j in range(sweep):
        if j%50 == 0: print('{} and {}'.format(j,k))
        
        par.tau_m = tau[j]
        
        '--------'

        neuron = models.NeuronClass(par)
        loss = nn.MSELoss(reduction='sum')
        w_0 = .02
        par.bound = 'False'
        par.optimizer = 'online'
        neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
        
        neuron.state()
        for t in range(par.T):       
            'online update'
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online()    
            'update state variables'        
            neuron(x_data[:,t])        
        
        w1_on[j,k] = neuron.w[0].item()
        w2_on[j,k] = neuron.w[1].item()
            
        '--------'    
        
        neuron = models.NeuronClass(par)
        loss = nn.MSELoss(reduction='sum')
        w_0 = .02
        par.bound = 'False'
        par.optimizer = 'SGD'
        neuron.w = nn.Parameter(w_0*torch.ones(par.N)).to(par.device)
        optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
        
        v = []
        neuron.state()
        for t in range(par.T):  
            v.append(neuron.v)       
            neuron(x_data[:,t])    
            
        x_hat = torch.einsum("bt,j->btj",torch.stack(v,dim=1),neuron.w)
        E = .5*loss(x_hat,x_data)
        
        optimizer.zero_grad()
        E.backward()
        optimizer.step()
        
        w1_sgd[j,k] = neuron.w[0].item()
        w2_sgd[j,k] = neuron.w[1].item()
    
savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/suppfig_timescale_separation/'
np.save(savedir+'w1_on',w1_on)
np.save(savedir+'w1_sgd',w1_sgd)
np.save(savedir+'w2_on',w2_on)
np.save(savedir+'w2_sgd',w2_sgd)
np.save(savedir+'tau',tau)
np.save(savedir+'eta',eta)

'plot'
w1_on = np.load(savedir+'w1_on.npy')
w1_sgd = np.load(savedir+'w1_sgd.npy')
w2_on = np.load(savedir+'w2_on.npy')
w2_sgd = np.load(savedir+'w2_sgd.npy')

from matplotlib.colors import LogNorm
class MidPointLogNorm(LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        LogNorm.__init__(self,vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint=midpoint
    def __call__(self, value, clip=None):
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(np.log(value), x, y))

hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
            
test = np.abs(w1_sgd-w1_on)**2/w1_sgd
test[test==0]=10e-8
test[test>10e16]=10e16

fig = plt.figure(figsize=(7,6), dpi=300)
plt.pcolormesh(eta,1/tau,test,norm=MidPointLogNorm(midpoint=1e1,vmax=1e16),cmap=funs.get_continuous_cmap(hex_list))
plt.xscale('log')
plt.yscale('log')
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$\eta$')
plt.ylabel(r'$1/\tau_m$')
plt.savefig(savedir+'eta_tau_comparison.png',format='png', dpi=300)
plt.savefig(savedir+'eta_tau_comparison.pdf',format='pdf', dpi=300)
plt.close('all')