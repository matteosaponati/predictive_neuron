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
import numpy as np
import matplotlib.pyplot as plt

'set parameters'
par = types.SimpleNamespace()
par.sequence = "deterministic"
par.dt = .05
par.Dt = 2
par.N_seq = 100
par.N_dist = 100
par.offset = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.batch = 1
par.T = 2*int((par.Dt*par.N_seq)/(par.dt))
par.device = 'cpu'
savedir = '/Users/saponatim/Desktop/'

'set total input'
par.N = par.N_seq+par.N_dist   

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
plt.savefig(savedir+'sequence_example.png',format='png', dpi=300)
plt.savefig(savedir+'sequence_example.pdf',format='pdf', dpi=300)
plt.close('all') 

'create firing rate'
par.tau_m = 10



