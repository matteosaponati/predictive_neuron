"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_nevian2006.py"
'Nevian et al (2006) Spine Ca2+ signaling in spike-timing-dependent plasticity
Journal of Neuroscience'

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

'---------------------------------------------'
def train_stdp(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, _, _, _ = funs_train.forward_NumPy(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print('pairing protocol {} out of {}'.format(e,par.epochs))        
    return w1, w2
'---------------------------------------------'

'set model'
par = types.SimpleNamespace()
par.dt = .05
par.eta = 2e-4
par.tau_m = 25.
par.v_th = 2.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N  = 2
par.T = int(300/par.dt)
par.epochs = 60

'set initial conditions'
w_0 = np.array([.01,.08])

"""
we reproduce the experimental protocol by increasing the frequency of post bursts
inputs:
    1. n_spk: total number of post spikes in the bursts
    2. dt_burst, dt: delay between post spikes, delay between pre and first post
"""
n_spikes = 3
dt_burst, dt = (np.array([10.,20.,50.])/par.dt).astype(int), int(10./par.dt)

w_pre,w_post = [],[]
for j in dt_burst:
    
    timing_pre = [np.array(0),dt+np.arange(0,j*n_spikes,j)]
    timing_post = [np.arange(0,j*n_spikes,j),np.array(np.arange(0,j*n_spikes,j)[-1]+ dt)] 
    
    'pre-post protocol'
    x_data = funs.get_sequence_stdp(par,timing_pre)
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2 = train_stdp(par,neuron,x_data)
    w_pre.append(w1[-1])

    'post-pre protocol'  
    x_data = funs.get_sequence_stdp(par,timing_post)
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()[::-1]
    w1,w2 = train_stdp(par,neuron,x_data)
    w_post.append(w2[-1])

'plot'

fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(1e3/(dt_burst[::-1]*par.dt),np.array(w_pre)[::-1]/w_0[0],color='royalblue',linewidth=2,label='pre-post')
plt.plot(1e3/(dt_burst[::-1]*par.dt),np.array(w_post)[::-1]/w_0[0],color='rebeccapurple',linewidth=2,label='post-pre')
'add experimental data'
x = [20,50,100]
y_pre, y_pre_e = [1.1,2,2.25],[.3,.3,.6]
plt.scatter(x,y_pre,color='k',s=20)
plt.errorbar(x,y_pre,yerr = y_pre_e,color='k',linestyle='None')
y_post, y_post_e = [.74,.74,.55],[.2,.1,.15]
plt.scatter(x,y_post,color='k',s=20)
plt.errorbar(x,y_post,yerr = y_post_e,color='k',linestyle='None')
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.xlabel(r'frequency [Hz]')
plt.ylabel(r'$w/w_0$')
plt.savefig(os.getcwd()+'/plots/stdp_nevian2006.png', format='png', dpi=300)
# plt.savefig(os.getcwd()+'/plots/stdp_nevian2006.pdf', format='pdf', dpi=300)
plt.close('all')

"RMS error"
error_prepost = np.sqrt(np.sum((np.array(w_pre)/w_0[0] - np.array(y_pre))**2)/len(y_pre))
error_postpre = np.sqrt(np.sum((np.array(w_post)/w_0[0] - np.array(y_post))**2)/len(y_post))