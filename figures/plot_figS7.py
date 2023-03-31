import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

from utils.funs import get_dir_results, get_continuous_cmap
from utils.data import get_spike_times

'-----------------------------------------'

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'selforg'
par.network_type = 'all'
par.package = 'NumPy'

par.bound = 'none'
par.eta = 2e-6
par.batch = 1
par.epochs = 2000
    
par.init = 'fixed'
par.init_mean = .06
par.init_rec = .0003
    
par.Dt = 2
par.n_in = 2
par.nn = 8
par.delay = 8

par.freq = 5.
par.jitter = 1.

par.dt = .05
par.tau_m = 25.
par.v_th = 2.9
par.tau_x = 2.

par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
    
par.N = par.n_in+par.nn
    
par.dir_output = '../_results/'

'-----------------------------------------'

path = get_dir_results(par)

w = np.load(path+'w.npy')

hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(6,6), dpi=300)    

divnorm = colors.TwoSlopeNorm(vmin=w[-1,:].min(),vcenter=0, vmax=w[-1,:].max())
plt.imshow(w[-1,:],cmap=get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')

fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.title(r'$\vec{w}$')
plt.ylabel(r'inputs')
plt.xlabel(r'neurons')
plt.savefig('plots/figS7_c.pdf',format='pdf', dpi=300)
plt.close('all')

'-----------------------------------------'

"""
quantification of the number of neurons that needs to be activate such that
the network can recall the whole sequence. We show that the number of neurons 
required for the recall decreases consistently across epochs. 
"""

from models.SelfOrgNetworkClass import NetworkClassNumPy
from utils.TrainerClassNumPy_SelfOrg import TrainerClass

spk_times = get_spike_times(par)

def get_dataset_selforg(par,spk_times):

    x = np.zeros((par.batch,par.n_in,par.nn,par.T))

    spk_times = np.repeat(spk_times[np.newaxis,:],par.batch,axis=0)

    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.batch,par.n_in,par.nn))
        freq = np.repeat(freq[:,:,:,np.newaxis],par.T,axis=3)
        x[np.random.rand(par.batch,par.n_in,par.nn,par.T)<(freq*par.dt/1000)] = 1

    if par.jitter > 0.:
        jitter = np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,
                                        (par.batch,par.n_in))
        spk_times += np.repeat(jitter[:,:,np.newaxis],par.nn,axis=2)
    
    for b in range(par.batch):
        for n in range(par.n_in):
            for nn in range(par.nn):
                
                'add inputs only to selected neurons'
                if nn in range(par.subseq):    
                    x[b,n,nn,spk_times[b,n,nn]] = 1
                    x[b,n,nn,:] = np.convolve(x[b,n,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
    
    return x

'-----------------------------'
'before'
par.subseq, par.epochs = 2, 1
par.freq = 5.
par.jitter = 1.
x = get_dataset_selforg(par,spk_times)

train_data = get_dataset_selforg(par,spk_times)
test_data = get_dataset_selforg(par,spk_times)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[0,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _ = trainer._do_test()

zPlot = []
for n in range(par.nn):
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS7_b_before.pdf',format='pdf', dpi=300)
plt.close('all') 
 
'-----------------------------'
'learning'
par.subseq, par.epochs = par.nn, 1
x = get_dataset_selforg(par,spk_times)

train_data = get_dataset_selforg(par,spk_times)
test_data = get_dataset_selforg(par,spk_times)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[10,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _ = trainer._do_test()

zPlot = []
for n in range(par.nn):
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS7_b_learning.pdf',format='pdf', dpi=300)
plt.close('all') 

'-----------------------------'
'after'
par.subseq, par.epochs = 2, 1
x = get_dataset_selforg(par,spk_times)

train_data = get_dataset_selforg(par,spk_times)
test_data = get_dataset_selforg(par,spk_times)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[-1,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _ = trainer._do_test()

zPlot = []
for n in range(par.nn):
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS7_b_after.pdf',format='pdf', dpi=300)
plt.close('all') 

'-----------------------------'
'after spontaneous'
par.subseq, par.epochs = 2, 1
x = get_dataset_selforg(par,spk_times)

train_data = get_dataset_selforg(par,spk_times)
test_data = get_dataset_selforg(par,spk_times)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[-1,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _ = trainer._do_test()

zPlot = []
for n in range(par.nn):
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in range(par.nn):
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS7_b_spontaneous.pdf',format='pdf', dpi=300)
plt.close('all') 