import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from models.NeuronClass import NeuronClassNumPy
from utils.TrainerClassNumPy import TrainerClass
from utils.data import get_spike_times, get_dataset_sequence

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'sequence'
par.package = 'NumPy'

par.bound = 'soft'
par.eta = 5e-4
par.batch = 1
par.epochs = 300
    
par.sequence = 'deterministic'
par.Dt = 4
par.N_seq = 2
par.N_dist = 0
par.N = par.N_seq+par.N_dist

par.freq = 0.
par.jitter = 0.
par.onset = 0

par.dt = .05
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

par.T = int(100/par.dt)

spk_times = get_spike_times(par)
x,onsets = get_dataset_sequence(par,spk_times)

par.train_nb = par.batch
par.test_nb = par.batch

vList = np.zeros((3,par.epochs,par.batch,par.T))
wList = np.zeros((3,par.epochs,par.N))
zList = np.zeros((3,par.epochs,par.batch,par.T))

par.init = 'fixed'
par.init_mean = 0.

w0_list = [[.15,.15,],[.06,.06],[.02,.15]]

for idx, w0 in enumerate(w0_list):

    print(idx)
    neuron = NeuronClassNumPy(par)
    neuron.initialize()
    neuron.w = np.array(w0)

    trainer = TrainerClass(par,neuron,x,x)

    for e in range (par.epochs):

        trainer._do_train()
        _, v, z, _, _ = trainer._do_test()

        vList[idx,e,:] = v 
        zList[idx,e,:] = z[0] 
        wList[idx,e,:] = trainer.neuron.w

        if e%50 == 0: print(e)

'-------------------------------'

'plots'

c=['mediumvioletred','mediumslateblue','lightseagreen','salmon']

zPlot = []
for s in range(len(w0_list)):
    zPlot.append([])
    for e in range(par.epochs):
        zPlot[s].append((np.where(zList[s,e,0,:])[0]*par.dt).tolist())

fig = plt.figure(figsize=(6,6), dpi=300)
for s in range(3):
    for k,j in zip(zPlot[s],range(par.epochs)):
        plt.scatter([j]*len(k),k,c=c[s],s=7)
plt.ylabel(r'output spikes (s) [ms]')
for k in spk_times*par.dt:
    plt.axhline(y=k,color='k',linewidth=1.5,linestyle='dashed')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,10)
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('plots/figS1_bleft.pdf',format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(6,6), dpi=300)
for s in range(wList.shape[0]):
    plt.plot(wList[s,:,0],color=c[s],linewidth=2)
    plt.plot(wList[s,:,1],color=c[s],linewidth=2,linestyle='dashed')
plt.xlabel('epochs')
plt.ylabel(r'synaptic weights $\vec{w}$')
plt.xlim(0,par.epochs)
plt.ylim(bottom=0)
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('plots/figS1_bright.pdf',format='pdf', dpi=300)
plt.close('all')

'----------------------------------'

par.eta = 1e-4
par.epochs = 10

w_sweep = np.arange(.01,.2,.01)

par.init_mean = 0.
w1 = np.zeros((w_sweep.size,w_sweep.size))
w2 = np.zeros((w_sweep.size,w_sweep.size))

for i, w_i in enumerate(w_sweep):
    for j, w_j in enumerate(w_sweep):

        neuron = NeuronClassNumPy(par)
        neuron.initialize()
        
        neuron.w = np.array([w_i,w_j])

        trainer = TrainerClass(par,neuron,x,x)

        for e in range (par.epochs):
            trainer._do_train()

        w1[i,j] = trainer.neuron.w[0]
        w2[i,j] = trainer.neuron.w[1]

dw_1 = w1 - np.tile(w_sweep,(w_sweep.size,1)).T
dw_2 = w2 - np.tile(w_sweep,(w_sweep.size,1))

fig = plt.figure(figsize=(7,7), dpi=300)
plt.xlabel(r'$w_{2}$')
plt.ylabel(r'$w_{1}$')
plt.xlim(-.5,w_sweep.size-.5)
plt.ylim(-.5,w_sweep.size-.5)
plt.xticks(np.arange(w_sweep.size)[::8],np.round(w_sweep,2)[::8])
plt.yticks(np.arange(w_sweep.size)[::8],np.round(w_sweep,2)[::8])
x,y = np.meshgrid(np.arange(len(w_sweep)),np.arange(len(w_sweep)))
plt.quiver(x,y,dw_2,dw_1,angles='xy',color='mediumvioletred',headwidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('plots/figS1_a.pdf', format='pdf', dpi=300)
plt.close('all')    

'----------------------------------'