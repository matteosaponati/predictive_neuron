"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"input_sequences_capacity_example.py":
    
example of input sequence

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

par = types.SimpleNamespace()

'set model'
par.name = 'multisequence'
par.dt = .05
par.eta = 2e-7
par.tau_m = 18.
par.v_th = 2.4
par.tau_x = 2.

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.N_sub = 8
par.delay = 20
par.batch = 3
par.N = par.N_sub*par.batch 
par.N_subseq = [np.arange(k,k+par.N_sub) 
                    for k in np.arange(0,par.N+par.N_sub,par.N_sub)]
timing = []
for b in range(par.batch):
    timing.append(((np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub) + b*(par.Dt*par.N_sub + par.delay))/par.dt).astype(int))

'set training algorithm'
par.bound = 'none'
par.epochs = 2000

'set initialization'
par.init = 'fixed'
par.init_mean = 0.04
par.init_a, par.init_b = 0, .06

'set noise sources'
par.noise = 1
par.freq_noise = 1
par.freq = 5
par.jitter_noise = 1
par.jitter = 2

par.T = int((par.Dt*par.N + (par.batch)*par.delay +par.jitter)/(par.dt))

'make figure'
fig = plt.figure(figsize=(4,4), dpi=300)
offset = np.random.permutation(par.N)
for k in range(par.N):
    'set background'    
    bg = (np.where(np.random.rand(par.T)<(np.random.uniform(0,(par.freq*par.dt)/1000)))[0]).astype(int)
    plt.eventplot(bg,lineoffsets = offset[k],linelengths=3,linewidths=.5,alpha=.5,colors = 'grey')
    'set subsequences'
    plt.eventplot([np.array(timing).flatten()[k]],lineoffsets = offset[k],linelengths=3,linewidths=.5,colors = 'purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(os.getcwd()+'/plots/sequence_capacity_example.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/sequence_capacity_example.pdf',format='pdf', dpi=300)
plt.close('all') 