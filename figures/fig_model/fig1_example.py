"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_example.py":
reproduce the dynamics of anticipation and predictive plasticity of Figure 1
    
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
par.eta = 5e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.noise = False
par.N = 2
par.T = int(100/par.dt)
par.epochs = 300
timing = (np.array([2.,6.])/par.dt).astype(int)
x_data = funs.get_sequence_stdp(par,timing)

"""
dynamics of anticipation and predictive plasticity
inputs:
    1. w_0: different set of initial conditions
"""
w_0 = [.005,.03,.05]
   
w1_tot, w2_tot = [],[]
v_tot, spk_tot = [], []    
for k in range(len(w_0)):
    
    'numerical solution'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0[k]*np.ones(par.N)
    
    w,v,spk,_ = funs_train.train_NumPy(par,neuron,x=x_data)
    
    w1_tot.append(np.vstack(w)[:,0])
    w2_tot.append(np.vstack(w)[:,1])
    v_tot.append(v)
    spk_tot.append(spk)
    
'plots'
c=['mediumvioletred','mediumslateblue','lightseagreen','salmon']

fig = plt.figure(figsize=(6,6), dpi=300)
for s in range(len(spk_tot)):
    for k,j in zip(spk_tot[s],range(par.epochs)):
        plt.scatter([j]*len(k),k,c=c[s],s=7)
plt.ylabel(r'output spikes (s) [ms]')
for k in timing*par.dt:
    plt.axhline(y=k,color='k',linewidth=1.5,linestyle='dashed')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,10)
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('s_convergence.png',format='png', dpi=300)
plt.savefig('s_convergence.pdf',format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(6,6), dpi=300)
for s in range(len(w_0)):
    plt.plot(w1_tot[s],color=c[s],linewidth=2)
    plt.plot(w2_tot[s],color=c[s],linewidth=2,linestyle='dashed')
plt.xlabel(r'epochs')
plt.ylabel(r'synaptic weights $\vec{w}$')
plt.xlim(0,par.epochs)
plt.ylim(bottom=0)
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('w_convergence.png',format='png', dpi=300)
plt.savefig('w_convergence.pdf',format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(v_tot[2][0],color='navy',linewidth=2,label='epoch 0')
plt.plot(v_tot[2][100],color='blue',linewidth=2,label='epoch 100')
plt.legend()
plt.xlabel(r'time')

'----------'
'----------'

"""
dynamics in the parameter space
inputs:
    1. w_sweep: set of initial conditions to sample parameter space
"""
w_sweep = np.arange(.01,.15,.01)
par.eta = 1e-4
par.epochs = 10

'parameter space'
w_1,w_2 = np.zeros((len(w_sweep),len(w_sweep))), np.zeros((len(w_sweep),len(w_sweep)))
for k in range(len(w_sweep)):
    for j in range(len(w_sweep)):
        'numerical solution'
        neuron = models.NeuronClass_NumPy(par)
        neuron.w = np.array([w_sweep[k],w_sweep[j]])
        w,v,spk,_ = funs_train.train_NumPy(par,neuron,x_data)
        w_1[k,j] = w[-1][0]
        w_2[k,j] = w[-1][1]
        
dw_1, dw_2 = w_1 - np.tile(w_sweep,(len(w_sweep),1)).T, w_2 - np.tile(w_sweep,(len(w_sweep),1))

'plot'
fig = plt.figure(figsize=(7,7), dpi=300)
plt.xlabel(r'$w_{2}$')
plt.ylabel(r'$w_{1}$')
'check how to put x and y ticks'
plt.xlim(-.5,len(w_sweep)-.5)
plt.ylim(-.5,len(w_sweep)-.5)
plt.xticks(np.arange(len(w_sweep))[::8],np.round(w_sweep,2)[::8])
plt.yticks(np.arange(len(w_sweep))[::8],np.round(w_sweep,2)[::8])
'plot vector field'
x,y = np.meshgrid(np.arange(len(w_sweep)),np.arange(len(w_sweep)))
plt.quiver(x,y,dw_2,dw_1,angles='xy',color='mediumvioletred',headwidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('vector_field.png', format='png', dpi=300)
plt.savefig('vector_field.pdf', format='pdf', dpi=300)
plt.close('all')