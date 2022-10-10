"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_classical.py"
reproduce classical anti-symmetrical STDP windows with predictive plasticity

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 2e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 60

'initial conditions'
w_0 = np.array([.001,.11])

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
    2. tau_sweep: different values of the membrane time constant
"""
delay = (np.arange(2,50,5)/par.dt).astype(int)
tau_sweep = [10.,15.,20.]

w_prepost  = [[] for k in range(len(tau_sweep))]
w_postpre = [[] for k in range(len(tau_sweep))]

for k in range(len(tau_sweep)):
    
    'set membrane time constant'
    par.tau_m = tau_sweep[k]
    print('membrane time constant '+str(par.tau_m)+' ms')
    
    'pre-post protocol with different Dt'
    for d in delay:
        
        'set inputs'
        timing = np.array([0,0+d]).astype(int)
        x_data = funs.get_sequence_stdp(par,timing)
        
        'pre-post pairing'
        neuron = models.NeuronClass_NumPy(par)
        neuron.w = w_0.copy()
        w1,w2 = funs_train.train_stdp(par,neuron,x_data)
        w_prepost[k].append(w1[-1])
        
        'post-pre pairing'
        neuron = models.NeuronClass_NumPy(par)
        neuron.w = w_0[::-1].copy()
        w1,w2 = funs_train.train_stdp(par,neuron,x_data)
        w_postpre[k].append(w2[-1])  
    
'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
for k in range(len(tau_sweep)):
    plt.plot(-delay[::-1]*par.dt,w_prepost[k][::-1]/w_0[0],linewidth=2)
    plt.plot(delay*par.dt,w_postpre[k]/w_0[0],linewidth=2)
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.axvline(x=0, color='black',linewidth=1.5)
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.savefig('stdp_window.png', format='png', dpi=300)
plt.savefig('stdp_window.pdf', format='pdf', dpi=300)
plt.close('all')