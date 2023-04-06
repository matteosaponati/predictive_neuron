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
par.tau_m = 30.
par.v_th = 3.1
par.tau_x = 2.

par.rep = 3

par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                        par.jitter + 80)/(par.dt))
    
par.N_in = par.n_in*par.nn
par.N = par.N_in+par.nn
    
par.dir_output = '../_results/'

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


'-----------------------------------------'

'compute the percentage of succesfull sequence retrieval for a given number of random afferents'

performance = np.zeros((7,20))
timing = np.zeros((7,20))

for i, par.n_afferents in enumerate(range(1,8)):
    
    for r, par.rep in enumerate(range(20)):
        
        path = get_dir_results(par)
        
        w = np.load(path+'w.npy')
        mask = np.load(path+'mask.npy')
        
        '---------'
        
        par.subseq, par.epochs = 2, 1
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
        
        if len(np.where(z[0][0,:,:])[1]) >0:

            first_spk = np.where(z[0][0,:,:])[1].min()*par.dt
            last_spk = np.where(z[0][0,:,:])[1].max()*par.dt
        
            timing[i,r] = last_spk-first_spk
            
            performance[i,r] = np.where(z[0][0,:,:])[0].size

performance /= par.nn

np.save('plots/timing',timing)
np.save('plots/performance',performance)

fig = plt.figure(figsize=(4,6),dpi=300)
plt.plot(range(1,8),timing.mean(axis=1),color='purple')
plt.fill_between(range(1,8),timing.mean(axis=1)-stats.sem(timing,axis=1),
                 timing.mean(axis=1)+stats.sem(timing,axis=1),
                 color='purple',alpha=.3)
plt.xlabel('random input subsets [count]')
plt.ylabel(r'$\Delta t$ [ms]')
plt.savefig('plots/timing.pdf',format='pdf',dpi=300)
plt.close()

fig = plt.figure(figsize=(4,6),dpi=300)
plt.plot(range(1,8),performance.mean(axis=1),color='purple')
plt.fill_between(range(1,8),performance.mean(axis=1)-stats.sem(performance,axis=1),
                 performance.mean(axis=1)+stats.sem(performance,axis=1),
                 color='purple',alpha=.3)
plt.xlabel('random input subsets [count]')
plt.ylabel('number of recalled neurons [%]')
plt.savefig('plots/performance.pdf',format='pdf',dpi=300)