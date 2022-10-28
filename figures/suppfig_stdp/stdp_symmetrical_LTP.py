"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_symmetrical.py"
reproduce symmetrical STDP windows with predictive plasticity

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import os
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 4e-4
par.tau_m = 5.
par.v_th = .4
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 60

'initial conditions'
w_0 = np.array([.0007,.02])

'---------------------------------------------'
def train_stdp(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, _, _, _ = funs_train.forward_NumPy(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print(e)        
    return w1, w2
'---------------------------------------------'

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
"""
delay = (np.arange(2,50,10)/par.dt).astype(int)

w_prepost,w_postpre = [],[]
for k in range(len(delay)):
    
    'set inputs'
    timing = np.array([0,0+ delay[k]]).astype(int)
    x_data = funs.get_sequence_stdp(par,timing)
    
    'pre-post pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2 = train_stdp(par,neuron,x_data)
    w_prepost.append(w1[-1])
    
    'post-pre pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0[::-1].copy()
    w1,w2 = train_stdp(par,neuron,x_data)
    w_postpre.append(w2[-1])  
    
'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(-delay[::-1]*par.dt,w_prepost[::-1]/w_0[0],linewidth=2,color='mediumvioletred')
plt.plot(delay*par.dt,w_postpre/w_0[0],linewidth=2,color='mediumvioletred')
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig(os.getcwd()+'/plots/stdp_window_symmetrical.png', format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/stdp_window_symmetrical.pdf', format='pdf', dpi=300)
plt.close('all')