"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"input_sequence_example.py"
example of input sequence - Fig 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import types
import os
import numpy as np
import matplotlib.pyplot as plt

par = types.SimpleNamespace()

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_seq = 100
par.N_dist = 100
par.N = par.N_seq+par.N_dist   
timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)

par.dt = .05
par.tau_x = 2.

'set noise sources'
par.noise = 1
par.freq_noise = 1
par.freq = 10
par.jitter_noise = 1
par.jitter = 2
par.T = int(2*(par.Dt*par.N_seq + par.jitter)/par.dt) 
par.onset = 1

'---------------------------------------------'

'create sequence'
if par.sequence == 'deterministic':
    timing = np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt
if par.sequence == 'random':
    timing = np.cumsum(np.random.randint(0,par.Dt,par.N_seq))/par.dt
'set offset and jitter'
timing += np.random.randint(0,par.T/2) + np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,par.N_seq)

'make figure'
fig = plt.figure(figsize=(3,3), dpi=300)
offset = np.random.permutation(par.N_seq+par.N_dist)
for k in range(par.N):
    'set background'    
    bg = (np.where(np.random.rand(par.T)<(np.random.uniform(0,par.freq*par.dt)))[0]).astype(int)
    plt.eventplot(bg,lineoffsets = offset[k],linelengths=3,linewidths=.5,alpha=.5,colors = 'grey')
    if k < par.N_seq: plt.eventplot([timing[k]],lineoffsets = offset[k],
                                         linelengths=3,linewidths=.5,colors = 'purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(os.getcwd()+'/plots/sequence_example.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/sequence_example.pdf',format='pdf', dpi=300)
plt.close('all') 