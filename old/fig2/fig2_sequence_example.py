"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig2_sequence_example.py":
    
    - example of input sequence
    - example of spike output
    - example fo weights dynamics

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import types
import numpy as np
#import tools
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs


par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.
'architecture'
par.N = 100
par.Dt = 4
par.T = int((par.Dt*par.N*2)/(par.dt))
par.batch = 1
par.device = 'cpu'

par.freq = .01
par.jitter = 2

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'
savedir = '/gs/home/saponatim/'

'1. sequence example'
import matplotlib.pyplot as plt
def sequence_example(par,savedir):
    
    '1. sequence example'

    fig = plt.figure(figsize=(4,4), dpi=300)
    timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
    timing += np.random.randint(0,par.T/2) + np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,par.N)
    offset = 1
    for k in range(par.N):
        bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
        plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
        plt.eventplot([timing[k]],lineoffsets = offset,linelengths = 3,linewidths = .5,colors = 'purple')
        offset += 1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'spk_volley.png',format='png', dpi=300)
    plt.savefig(savedir+'spk_volley.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
sequence_example(par,savedir)

type = 'sequence'
w = np.load(par.dir+'w_{}.npy'.format(type))
loss = np.load(par.dir+'loss_{}.npy'.format(type))
v = np.load(par.dir+'v_{}.npy'.format(type))
spk = np.load(par.dir+'spk_{}.npy'.format(type),allow_pickle=True)


' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_sequence.png',format='png', dpi=300)
plt.savefig('w_sequence.pdf',format='pdf', dpi=300)
plt.close('all')


'v dynamics'
v_spike = v[0,0,:].copy()
v_spike[v_spike>2]=7
fig = plt.figure(figsize=(4,4), dpi=300)
plt.plot(v_spike,linewidth=2,color='navy')
#plt.xlim(0,par.T*par.dt)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('v')
plt.savefig(savedir+'v_epoch_0.png',format='png', dpi=300)
plt.savefig(savedir+'v_epoch_0.pdf',format='pdf', dpi=300)
plt.close('all') 

v_spike = v[-1,0,:].copy()
v_spike[v_spike>2]=7
fig = plt.figure(figsize=(4,4), dpi=300)
plt.plot(v_spike,linewidth=2,color='navy')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('v')
plt.savefig(savedir+'v_epoch_fin.png',format='png', dpi=300)
plt.savefig(savedir+'v_epoch_fin.pdf',format='pdf', dpi=300)
plt.close('all') 


'PATTERN'

' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_patterm.png',format='png', dpi=300)
plt.savefig('w_pattern.pdf',format='pdf', dpi=300)
plt.close('all')
