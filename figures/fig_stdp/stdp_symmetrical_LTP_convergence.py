"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_symmetrical_LTP_convergence.py":
reproduce symmetrical STDP windows with predictive plasticity
convergence to anticipatory firing
    
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
par.eta = 1e-5
par.tau_m = 5.
par.v_th = .4
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(200/par.dt)
par.epochs = 2000

'initial conditions'
w_0 = np.array([.0007,.02])

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs, asymptotic dynamics
inputs:
    1. delay: range of delay considered
    2. tau_sweep: different values of the membrane time constant
"""

delay = (np.array([4,10])/par.dt).astype(int)

w1_prepost   = []
w2_prepost   = []
w1_postpre   = []
w2_postpre   = []
spk_prepost  = []
spk_postpre  = []

for d in delay:
    
    'set inputs'
    timing = np.array([0,0+d]).astype(int)
    x_data = funs.get_sequence_stdp(par,timing)
    
    'pre-post pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2,v,spk,_ = funs_train.train_NumPy(par,neuron,x_data)
    spk_prepost.append(spk)
    w1_prepost.append(w1)
    w2_prepost.append(w2)
    
    'post-pre pairing'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0[::-1].copy()
    w1,w2,v,spk,_ = funs_train.train_NumPy(par,neuron,x_data)
    spk_postpre.append(spk)
    w1_postpre.append(w1)
    w2_postpre.append(w2)
        
'plot'
c=['mediumvioletred','mediumslateblue','lightseagreen','salmon']

'plot'
fig = plt.figure(figsize=(7,7), dpi=300)
plt.xlabel(r'$w_{2}$')
plt.ylabel(r'$w_{1}$')
for d in range(len(delay)):
    plt.plot(w2_prepost[d],w1_prepost[d],linewidth=2,color=c[d])
    plt.plot(w2_postpre[d],w1_postpre[d],linewidth=2,linestyle='dashed',color=c[d])
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('stdp_classical_convergence.png', format='png', dpi=300)
plt.savefig('stdp_classical_convergence.png', format='pdf', dpi=300)
plt.close('all')
    