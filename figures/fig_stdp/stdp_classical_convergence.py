"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_classical_convergence.py":
reproduce classical anti-symmetrical STDP windows with predictive plasticity
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

from predictive_neuron import models, funs

'-----'
def train_STDP(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, _, spk, _ = funs.forward_NumPy(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print(e)        
    return w1, w2, spk

'-----'

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
par.epochs = 1000

'initial conditions'
w_0 = np.array([.001,.11])

"""
we reproduce the classical pre-post pairing protocol by changing the delay
between the two pre-synaptic inputs, asymptotic dynamics
inputs:
    1. delay: range of delay considered
    2. tau_sweep: different values of the membrane time constant
"""

delay = (np.array([4,8,10,20])/par.dt).astype(int)
tau_sweep = [10.,15.,20.]

w1_prepost   = [[] for k in range(len(tau_sweep))]
w2_prepost   = [[] for k in range(len(tau_sweep))]
w1_postpre   = [[] for k in range(len(tau_sweep))]
w2_postpre   = [[] for k in range(len(tau_sweep))]
spk_prepost = [[] for k in range(len(tau_sweep))]
spk_postpre = [[] for k in range(len(tau_sweep))]

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
        w1,w2,spk = train_STDP(par,neuron,x_data)
        spk_prepost[k].append(spk)
        w1_prepost[k].append(w1)
        w2_prepost[k].append(w2)
        
        'post-pre pairing'
        neuron = models.NeuronClass_NumPy(par)
        neuron.w = w_0[::-1].copy()
        w1,w2,spk = train_STDP(par,neuron,x_data)
        spk_postpre[k].append(spk)
        w1_postpre[k].append(w1)
        w2_postpre[k].append(w2)
        
'plot'
c=['mediumvioletred','mediumslateblue','lightseagreen','salmon']

'plot'
for t in range(len(tau_sweep)):
    
    fig = plt.figure(figsize=(7,7), dpi=300)
    plt.xlabel(r'$w_{2}$')
    plt.ylabel(r'$w_{1}$')
    for d in range(len(delay)):
        plt.plot(w2_prepost[t][d],w1_prepost[t][d],linewidth=2,color=c[d],label=r'$\Delta t$ = {}'.format(delay[d]*par.dt))
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.legend()
    plt.savefig('stdp_classical_convergence_LTP_tau_{}.png'.format(tau_sweep[t]), format='png', dpi=300)
#    plt.savefig('stdp_classical_convergence_LTP_tau_{}.pdf'.format(tau_sweep[t]), format='pdf', dpi=300)
    plt.close('all')
    
    fig = plt.figure(figsize=(7,7), dpi=300)
    plt.xlabel(r'$w_{2}$')
    plt.ylabel(r'$w_{1}$')
    for d in range(len(delay)):
        plt.plot(w2_postpre[t][d],w1_postpre[t][d],linewidth=2,color=c[d],label=r'$\Delta t$ = {}'.format(delay[d]*par.dt))
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.legend()
    plt.savefig('stdp_classical_convergence_LTD_tau_{}.png'.format(tau_sweep[t]), format='png', dpi=300)
    plt.savefig('stdp_classical_convergence_LTD_tau_{}.pdf'.format(tau_sweep[t]), format='pdf', dpi=300)
    plt.close('all')
    
    fig = plt.figure(figsize=(6,6), dpi=300)
    for d in range(len(delay)):
        for k,j in zip(spk_prepost[t][d],range(par.epochs)):
            plt.scatter([j]*len(k),k,c=c[d],s=7)
    plt.ylabel(r'output spikes (s) [ms]')
    plt.xlabel(r'epochs')
    plt.xlim(0,par.epochs)
    plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig('s_convergencee_LTP_tau_{}.png'.format(tau_sweep[t]),format='png', dpi=300)
    plt.savefig('s_convergence_LTD_tau_{}.pdf'.format(tau_sweep[t]),format='pdf', dpi=300)
    plt.close('all')
    
    fig = plt.figure(figsize=(6,6), dpi=300)
    for d in range(len(delay)):
        for k,j in zip(spk_postpre[t][d],range(par.epochs)):
            plt.scatter([j]*len(k),k,c=c[d],s=7)
    plt.ylabel(r'output spikes (s) [ms]')
    plt.xlabel(r'epochs')
    plt.xlim(0,par.epochs)
    plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig('s_convergencee_LTD_tau_{}.png'.format(tau_sweep[t]),format='png', dpi=300)
    plt.close('all')