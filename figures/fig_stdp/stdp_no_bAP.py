"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_no_bAP.py"
'Sjöström et al (2001) Rate, timing, and cooperativity jointly determine cortical 
synaptic plasticity. Neuron' 

model prediction that STDP-like learning window can be obtained even without
backpropagating action potential (bAP). For example, see:
- Lisman, J., & Spruston, N. (2005). Postsynaptic depolarization requirements 
for LTP and LTD: a critique of spike timing-dependent plasticity
Nature neuroscience
-Suvrathan, A. (2019). Beyond STDP—towards diverse and functionally relevant 
plasticity rules. Current opinion in neurobiology

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
par.eta = 1e-4
par.tau_m = 30.
par.v_th = 5.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 60

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs
inputs:
    1. delay: range of delay considered
"""
w_0 = np.array([.005,.05])
delay = (np.arange(0,50,10)/par.dt).astype(int)

w_prepost1,w_postpre1 = [],[]
w_prepost2,w_postpre2 = [],[]
for k in range(len(delay)):
    
    'set inputs'
    timing = np.array([2,2+ delay[k]]).astype(int)
    x_data = funs.get_sequence_stdp(par,timing)
    
    'pre-post pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2 = funs.train(par,neuron,x_data)
    w_prepost1.append(w1[-1])
    w_prepost2.append(w2[-1])
    
    'post-pre pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0[::-1].copy()
    w1,w2 = funs.train(par,neuron,x_data)
    w_postpre1.append(w2[-1])  
    w_postpre2.append(w1[-1])
    
'plot'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(delay,np.array(w_postpre1)/w_0[0],color='mediumvioletred',linewidth=2)
plt.plot(-delay[::-1],np.array(w_prepost1)[::-1]/w_0[0],color='mediumvioletred',linewidth=2)   
plt.plot(delay,np.array(w_postpre2)/w_0[1],color='purple',linewidth=2)
plt.plot(-delay[::-1],np.array(w_prepost2)[::-1]/w_0[1],color='purple',linewidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$\Delta t$ [ms]')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1,color='k',linestyle='dashed')
plt.savefig('stdp_no_bpAP.png',format='png', dpi=300)
plt.savefig('stdp_no_bpAP.pdf',format='pdf', dpi=300)
plt.close('all')
