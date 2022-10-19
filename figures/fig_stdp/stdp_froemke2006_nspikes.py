"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_froemke2006_frequency.py"
'Froemke et al (2006) Contribution of inidividual spikes in burst-induced 
long-term synaptic modification. Journal of Neuroscience'

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
par.eta = 8e-5
par.tau_m = 40.
par.v_th = 3.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(600/par.dt)
par.epochs = 30

'initial conditions'
w_0 = np.array([.14,.017])

"""
we reproduce the experimental protocol by increasing the number of inputs from
the second pre-synaptic neurons
input:
    
"""
n_spk = 5
dt_burst, dt = int(10/par.dt), int(5/par.dt)

w_post = []
for k in np.arange(1,n_spk+1):

    timing = [(np.arange(0,10*k,10)/par.dt).astype(int),dt]
    x_data = funs.get_sequence_stdp(par,timing)
 
    'postp-pre potocol'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2 = train_stdp(par,neuron,x_data)
    w_post.append(w2[-1])

'plot'
savedir = '/Users/saponatim/Desktop/'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(np.arange(1,n_spk+1),np.array(w_post)/w_0[1],color='rebeccapurple',linewidth=2)
'add experimental data'
x = [1,2,3,4,5]
y, y_e = [.7,.8,.9,1.02,1.2],[.1,.1,.1,.05,.05]
plt.scatter(x,y,color='k',s=20)
plt.errorbar(x,y,yerr = y_e,color='k',linestyle='None')
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.ylabel(r'$w/w_0$')
plt.xlabel(r'# spikes')
plt.xticks(np.arange(1,n_spk+1),np.arange(1,n_spk+1))
plt.ylim(.5,1.5)
# plt.savefig(os.getcwd()+'/plots/stdp_froemke2006_nspikes.pdf', format='pdf', dpi=300)
plt.savefig(os.getcwd()+'/plots/stdp_froemke2006_nspikes.png', format='png', dpi=300)
plt.close('all')

"RMS error"
error = np.sqrt(np.sum((np.array(w_post)/w_0[1] - np.array(y))**2)/len(y))