"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_initial_weight.py"
'Sjöström et al (2001) Rate, timing, and cooperativity jointly determine cortical 
synaptic plasticity. Neuron' 

reproduce how STDP potentiation depends on initial weight (Figure 5C)

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

from predictive_neuron import models, funs

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 1e-5
par.tau_m = 10.
par.v_th = 3.
par.tau_x = 2.
par.bound = 'none'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 20

"""
we qualitatively reproduce the experimental protocol by changing the initial
value of the synaptic weight while keeping the STDP protocol fixed
inputs:
    1. w_0_sweep: the different value of synaptic strength
"""
w_0_sweep = np.arange(.01,.06,.01)

'set inputs'
timing = (np.array([2.,6.])/par.dt).astype(int)
x_data = funs.get_sequence_stdp(par,timing)

w_pre = []
for k in range(len(w_0_sweep)):
    
    'pre-post pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = np.array([w_0_sweep[k],.13])
    w1,w2,v,spk = funs.train(par,neuron,x_data)
    w_pre.append(w1[-1])
    
'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(w_0_sweep,np.array(w_pre)/w_0_sweep,color='purple',linewidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$w_0$')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1,color='k',linestyle='dashed')
plt.savefig('stdp_initial_weight.png',format='png', dpi=300)
plt.savefig('stdp_initial_weight.pdf',format='pdf', dpi=300)
plt.close('all')