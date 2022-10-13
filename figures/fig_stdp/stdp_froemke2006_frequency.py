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
        if e%10 == 0: print(e)        
    return w1, w2
'---------------------------------------------'

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 3.4e-5
par.tau_m = 16.
par.v_th = 2.2
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(500/par.dt)
par.epochs = 40

'initial conditions'
w_0 = np.array([.12,.005])

"""
we reproduce the experimental protocol by increasing the pairing frequency
inputs:
    1. dt_burst, dt: delay between pairing, delay between pre and post (in ms)
"""
dt_burst, dt = (np.array([100.,20.,10.])/par.dt).astype(int) , int(6/par.dt)

w_post = []
for k in dt_burst:

    'set inputs'
    timing = [np.arange(0,k*5,k),np.arange(0,k*5,k)+dt]
    x_data = funs.get_sequence_stdp(par,timing)
    'numerical solutions'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2 = train_stdp(par,neuron,x_data)
    'get weights'
    w_post.append(w2[-1])

'plot'
savedir = '/Users/saponatim/Desktop/'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.scatter(1e3/(np.array(dt_burst)*par.dt),np.array(w_post)/w_0[1],color='rebeccapurple',s=40)
plt.plot(1e3/(np.array(dt_burst)*par.dt),np.array(w_post)/w_0[1],color='rebeccapurple',linewidth=2)
'add experimental data'
x = [10,50,100]
y, y_e = [.7,.99,1.3],[.05,.05,.1]
plt.scatter(x,y,color='k',s=20)
plt.errorbar(x,y,yerr = y_e,color='k',linestyle='None')
fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.xlabel(r'frequency [Hz]')
plt.ylabel(r'$w/w_0$')
plt.ylim(.5,1.5)
plt.savefig(savedir+'stdp_froemke2006_frequency.pdf', format='pdf', dpi=300)
plt.savefig(savedir+'stdp_froemke2006_frequency.png', format='png', dpi=300)
plt.close('all')

"RMS error"
error = np.sqrt(np.sum((np.array(w_post)/w_0[1] - np.array(y))**2)/len(y))