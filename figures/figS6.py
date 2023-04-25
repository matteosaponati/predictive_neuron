import argparse
import os
import numpy as np

from utils.funs import get_dir_results, get_hyperparameters
from models.SelfOrgNetworkClass import NetworkClassNumPy
from utils.TrainerClassNumPy_SelfOrg import TrainerClass
from utils.data import get_spike_times

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'selforg'
par.network_type = 'nearest'
par.package = 'NumPy'

par.bound = 'none'
par.eta = 8e-7
par.batch = 1
par.epochs = 1
    
par.init = 'fixed'
par.init_mean = .02
par.init_rec = .0003
    
par.Dt = 2
par.n_in = 8
par.nn = 10
par.delay = 4

par.freq = 5.
par.jitter = 1.

par.dt = .05
par.tau_m = 25.
par.v_th = 3.1
par.tau_x = 2.

par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                     par.jitter + 80)/(par.dt))
    
par.N = par.n_in+2
    
par.dir_output = '../_results/'

'---------------------------------------------'

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
                
                if n in range(par.subseq):
                    x[b,n,nn,spk_times[b,n,nn]] = 1
                x[b,n,nn,:] = np.convolve(x[b,n,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
    
    return x

'---------------------------------------------'

path = get_dir_results(par)

'get weights'
w = np.load(path+'w.npy')
'get spike stimes'
spk_times = get_spike_times(par)

rep = 1
n_required = np.zeros((w.shape[0],rep))

count = 0
for e in range(w.shape[0]):
    
    for k in range(rep):

        print(k)
            
        'run across neurons in the network'
        for par.subseq in range(1,par.nn+1):
        
            x = get_dataset_selforg(par,spk_times)

            par.train_nb = par.batch
            par.test_nb = par.batch
        
            network = NetworkClassNumPy(par)
            network.initialize()
            network.w = w[e,:].copy()

            trainer = TrainerClass(par,network,x,x)
            _, _, z, _, _ = trainer._do_test()
        
            '''
            ---------------------------------------------------------
            (1) span the spiking activity of every neuron in the network
            count if the neuron has been active during the simulation

            (2) check if every neuron in the network was active during 
            the simulation.

            if true, the current value of *subseq* represent the amount of input
            needed for the network to recall the whole sequence.
            if false, the number of input presented to the network is not 
            sufficient to trigger the recall of the whole sequence.
            '''

            count = 0
            for n in range(par.nn):
                if z[n][0] != []: count+=1

            if count == par.nn:
                n_required[e,k] = par.subseq
                break
            else:
                continue

'---------------------------------------'
'plot'

fig = plt.figure(figsize=(5,6), dpi=300)
plt.plot(n_required.mean(axis=1),linewidth=2,color='purple')
plt.fill_between(range(w.shape[0]),n_required.mean(axis=1)+n_required.std(axis=1),
                 n_required.mean(axis=1)-n_required.std(axis=1),
                 color='purple',alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel(r'# neurons for replay')
plt.savefig(os.getcwd()+'/plots/n_needed.png',format='png', dpi=300)
plt.savefig(os.getcwd()+'/plots/n_needed.pdf',format='pdf', dpi=300)
plt.close('all')           