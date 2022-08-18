"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"example_input.py"
create example input

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""
import torch
import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

'---------------------------------------'
'create parameter structure'
par = types.SimpleNamespace()
'architecture'
par.n_in = 26
par.nn = 8
par.batch = 1
par.lateral = 2
par.device = 'cpu'
par.dtype = torch.float
'model parameters'
par.dt = .05
par.tau_m = 15.
par.v_th = 3.5
par.tau_x = 2.
par.is_rec = True
par.online = True
'input'
par.delay = 4
par.jitter_noise = True
par.jitter = 2
par.fr_noise = True
par.freq = .01
par.w_0 = .02
par.T = int((par.n_in*par.delay+(par.n_in*par.Dt)+80)/par.dt)

par.loaddir = ''
par.savedir = ''

'--------------------'
'example inputs'

'define input sequence'
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)

'plot background'
fig = plt.figure(figsize=(8,4), dpi=300)
offset=1
for k in range(par.nn*par.n_in):
    bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
    plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
    offset += 1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.xlim(0,timing[-1][0][-1]+200)
plt.ylabel('inputs')
plt.savefig(par.savedir+'input_background.png',format='png', dpi=300)
plt.savefig(par.savedir+'input_background.pdf',format='pdf', dpi=300)
plt.close('all')

'plot pattern'
fig = plt.figure(figsize=(8,4), dpi=300)
offset=1
offset = 1
for n in range(par.nn):
    for k in range(par.n_in):        
        plt.eventplot([timing[n][0][k]+np.random.randint(-par.jitter,par.jitter)/par.dt],lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'purple')
        offset += 1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.xlim(0,timing[-1][0][-1]+200)
plt.ylabel('inputs')
plt.savefig(par.savedir+'input.png',format='png', dpi=300)
plt.savefig(par.savedir+'input.pdf',format='pdf', dpi=300)
plt.close('all')    