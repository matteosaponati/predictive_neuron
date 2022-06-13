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

savedir = '/gs/home/saponatim/'

''
dir = '/gs/home/saponatim/predictive_neuron/scripts/generalization/'



w = np.zeros((5000,100))
spk = []


for j in range(rep):
    'get weights'
    w[j,:] = np.load(dir+'w_bg_{}_rep_{}.npy'.format(jitter,j))    
    'get output'
    spk.append(np.load(dir+'spk_bg_{}_rep_{}.npy'.format(jitter,j),allow_pickle=True).tolist())
    
w = w[15,:,:]




w = np.load(dir+'w_pattern_offset_True.npy')
' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(np.sort(w,axis=1).T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_pattern.png',format='png', dpi=300)
plt.savefig('w_pattern.pdf',format='pdf', dpi=300)
plt.close('all')

spk = np.load(dir+'spk_pattern_offset_False.npy',allow_pickle=True).tolist()
for k,j in zip(spk,range(2000)):
    plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)

'normal'
dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/seq_generalization/'
jitter = .001
rep = 100

w = np.zeros((rep,5000,100))
spk = []

for j in range(rep):
    'get weights'
    w[j,:] = np.load(dir+'w_bg_{}_rep_{}.npy'.format(jitter,j))    
    'get output'
    spk.append(np.load(dir+'spk_bg_{}_rep_{}.npy'.format(jitter,j),allow_pickle=True).tolist())
    
w = w[15,:,:]

' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_seq.png',format='png', dpi=300)
plt.savefig('w_seq.pdf',format='pdf', dpi=300)
plt.close('all')

'4. output spikes'
fig = plt.figure(figsize=(6,6), dpi=300)
for k,j in zip(spk[15],range(5000)):
    plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.xlim(0,5000)
plt.ylim(0,200)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',color='darkgrey',linewidth=.7)
plt.savefig('spk_seq.png',format='png', dpi=300)
#plt.savefig('spk_seq_jitter.pdf',format='pdf', dpi=300)
plt.close('all')


'sequence jitter'
dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/seq_generalization/'

jitter = 6
rep = 100

w_tot = np.zeros((rep,5000,100))
spk = []

for j in range(rep):
    'get weights'
    w_tot[j,:] = np.load(dir+'w_jitter_{}_rep_{}.npy'.format(jitter,j))    
    'get output'
    spk.append(np.load(dir+'spk_jitter_{}_rep_{}.npy'.format(jitter,j),allow_pickle=True).tolist())
    
w = w_tot[14,:,:].copy()

' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_seq_jitter.png',format='png', dpi=300)
plt.savefig('w_seq_jitter.pdf',format='pdf', dpi=300)
plt.close('all')

'4. output spikes'
fig = plt.figure(figsize=(6,6), dpi=300)
for k,j in zip(spk[14],range(5000)):
    plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.xlim(0,5000)
plt.ylim(0,200)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',color='darkgrey',linewidth=.7)
plt.savefig('spk_seq_jitter.png',format='png', dpi=300)
#plt.savefig('spk_seq_jitter.pdf',format='pdf', dpi=300)
plt.close('all')


'sequence rate'

jitter = .01
rep = 100

w_rate = np.zeros((rep,5000,100))
spk_rate = []

for j in range(rep):
    'get weights'
    w_rate[j,:] = np.load(dir+'w_bg_{}_rep_{}.npy'.format(jitter,j))    
    'get output'
    spk_rate.append(np.load(dir+'spk_bg_{}_rep_{}.npy'.format(jitter,j),allow_pickle=True).tolist())
    
w = w_rate[2,:,:]

' weights dynamics'
hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig('w_seq_rate.png',format='png', dpi=300)
plt.savefig('w_seq_rate.pdf',format='pdf', dpi=300)
plt.close('all')

'4. output spikes'
fig = plt.figure(figsize=(6,6), dpi=300)
for k,j in zip(spk_rate[2],range(5000)):
    plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.xlim(0,5000)
plt.ylim(0,200)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',color='darkgrey',linewidth=.7)
plt.savefig('spk_seq_rate.png',format='png', dpi=300)
#plt.savefig('spk_seq_jitter.pdf',format='pdf', dpi=300)
plt.close('all')