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

from scipy import stats

'-----------------------------------------'

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'selforg'
par.network_type = 'random'
par.package = 'NumPy'

par.bound = 'none'
par.eta = 1e-6
par.batch = 1
par.epochs = 3000
    
par.init = 'random'
par.init_mean = .06
par.init_rec = .0003
    
par.Dt = 2
par.n_in = 2
par.nn = 8
par.delay = 8
par.n_afferents = 3

par.freq = 5.
par.jitter = 1.

par.dt = .05
par.tau_m = 25.
par.v_th = 3.1
par.tau_x = 2.

par.rep = 1

par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                        par.jitter + 80)/(par.dt))
    
par.N_in = par.n_in*par.nn
par.N = par.N_in+par.nn
    
par.dir_output = '../_results/'

'-----------------------------------------'

path = get_dir_results(par)

w = np.load(path+'w.npy')
mask = np.load(path+'mask.npy')

'-----------------------------------------'

"""
quantification of the number of neurons that needs to be activate such that
the network can recall the whole sequence. We show that the number of neurons 
required for the recall decreases consistently across epochs. 
"""

from models.SelfOrgNetworkClass import NetworkClassNumPy
from utils.TrainerClassNumPy_SelfOrg import TrainerClass

spk_times = get_spike_times(par)

def get_dataset_random(par,spk_times,mask):

    x = np.zeros((par.batch,par.N_in,par.nn,par.T))

    spk_times = np.repeat(spk_times[np.newaxis,:],par.batch,axis=0)

    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.batch,par.N_in,par.nn))
        freq = np.repeat(freq[:,:,:,np.newaxis],par.T,axis=3)
        x[np.random.rand(par.batch,par.N_in,par.nn,par.T)<(freq*par.dt/1000)] = 1

    if par.jitter > 0.:
        jitter = np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,
                                        (par.batch,par.N_in))
        spk_times += np.repeat(jitter[:,:,np.newaxis],par.nn,axis=2)
    
    for b in range(par.batch):

        if par.subseq == 1:
             
             for nn in range(par.nn):
                  for i in par.input_range:
                        if mask[i,nn] == True:
                       
                            x[b,i,nn,spk_times[b,0,nn]] = 1
                            x[b,i,nn,:] = np.convolve(x[b,i,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T] 

                            x[b,i+1,nn,spk_times[b,1,nn]] = 1
                            x[b,i+1,nn,:] = np.convolve(x[b,i+1,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T] 

        else:
            for n in range(par.N_in):
                for nn in range(par.nn):
                    x[b,n,nn,spk_times[b,n,nn]] = 1
                    x[b,n,nn,:] = np.convolve(x[b,n,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
            
    return x


'-----------------------------'

w = np.load(path+'w.npy')
mask = np.load(path+'mask.npy')

par.epochs = 1
par.subseq = par.N_in
par.input_range = [0,2]

train_data = get_dataset_random(par,spk_times,mask)
test_data = get_dataset_random(par,spk_times,mask)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[-1,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _, _ = trainer._do_test()

zList = []
for n in range(par.nn):
    
     if np.where(z[0][0][n,:]) != [] and len(np.where(z[0][0][n,:])[0]) > 0:
         zList.append(np.where(z[0][0][n,:])[0][0]*par.dt)
     
     else: 
         zList.append(par.T*par.dt)
zList = np.argsort(zList)

hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
divnorm = colors.TwoSlopeNorm(vmin=w[-1,:].min(),vcenter=0, vmax=w[-1,:].max())

fig = plt.figure(figsize=(6,6), dpi=300)    
plt.imshow(w[-1,:,zList].T,cmap=get_continuous_cmap(hex_list),norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.title(r'$\vec{w}$')
plt.ylabel(r'inputs')
plt.xlabel(r'neurons')
plt.savefig('plots/figS8_c.pdf',format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(6,6),dpi=300)
plt.imshow(w[0,:,zList].T,cmap=get_continuous_cmap(hex_list),norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0,0.01,1,0.97])
plt.colorbar()
plt.title(r'$\vec{w}$')
plt.ylabel(r'inputs')
plt.xlabel(r'neurons')
plt.savefig('plots/figS8_c_beginning.pdf',format='pdf',dpi=300)
plt.close('all')

'-----------------------------'

par.epochs = 1

'before'
par.subseq = 1
par.input_range = [0,2]
par.freq = 5.
par.jitter = 1.

train_data = get_dataset_random(par,spk_times,mask)
test_data = get_dataset_random(par,spk_times,mask)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[0,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _, _ = trainer._do_test()

zPlot = []
zList = []
for n in range(par.nn):
    
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())
     
     if np.where(z[0][0][n,:]) != [] and len(np.where(z[0][0][n,:])[0]) > 0:
         zList.append(np.where(z[0][0][n,:])[0][0]*par.dt)
     
     else: 
         zList.append(par.T*par.dt)
     
zList = np.argsort(zList)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in zList:
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS8_b_before.pdf',format='pdf', dpi=300)
plt.close('all') 
 
'-----------------------------'
'learning'
par.subseq = par.N_in
par.input_range = [0,2]

train_data = get_dataset_random(par,spk_times,mask)
test_data = get_dataset_random(par,spk_times,mask)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[10,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _, _ = trainer._do_test()

zPlot = []
zList = []
for n in range(par.nn):
    
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())
     
     if np.where(z[0][0][n,:]) != [] and len(np.where(z[0][0][n,:])[0]) > 0:
         zList.append(np.where(z[0][0][n,:])[0][0]*par.dt)
     
     else: 
         zList.append(par.T*par.dt)
     
zList = np.argsort(zList)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in zList:
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS8_b_learning.pdf',format='pdf', dpi=300)
plt.close('all') 

'-----------------------------'
'after'
par.subseq = 1
par.input_range = [0,2]

train_data = get_dataset_random(par,spk_times,mask)
test_data = get_dataset_random(par,spk_times,mask)
        
par.train_nb = par.batch
par.test_nb = par.batch
        
network = NetworkClassNumPy(par)
network.initialize()
network.w = w[-1,:]

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _, _ = trainer._do_test()

zPlot = []
zList = []
for n in range(par.nn):
    
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())
     
     if np.where(z[0][0][n,:]) != [] and len(np.where(z[0][0][n,:])[0]) > 0:
         zList.append(np.where(z[0][0][n,:])[0][0]*par.dt)
     
     else: 
         zList.append(par.T*par.dt)

zList = np.argsort(zList)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in zList:
    print(n)
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS8_b_after.pdf',format='pdf', dpi=300)
plt.close('all') 

'-----------------------------'
'after spontaneous'

train_data = get_dataset_random(par,spk_times,mask)
test_data = get_dataset_random(par,spk_times,mask)

trainer = TrainerClass(par,network,test_data,test_data)
_, _, z, _, _ = trainer._do_test()

zPlot = []
zList = []
for n in range(par.nn):
    
     zPlot.append((np.where(z[0][0][n,:])[0]*par.dt).tolist())
     
     if np.where(z[0][0][n,:]) != [] and len(np.where(z[0][0][n,:])[0]) > 0:
         zList.append(np.where(z[0][0][n,:])[0][0]*par.dt)
     
     else: 
         zList.append(par.T*par.dt)

zList = np.argsort(zList)

fig = plt.figure(figsize=(5,3), dpi=300)
m=1
for n in zList:
    plt.eventplot(zPlot[n],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.ylim(1,par.nn+1)
plt.xlim(0,par.T*par.dt)
plt.yticks(range(par.nn+2)[::2])
plt.xlabel('time [ms]')
plt.ylabel('neurons')
plt.savefig('plots/figS8_b_spontaneous.pdf',format='pdf', dpi=300)
plt.close('all') 