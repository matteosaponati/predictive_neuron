import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs, funs_train_inhibition

par = types.SimpleNamespace()

'training algorithm'
par.optimizer = 'Adam'
par.bound = 'None'
par.init = 'uniform'
par.init_mean = 0.02
par.init_a, par.init_b = 0, .03
par.epochs = 500
par.batch = 2
par.device = 'cpu'
par.dtype = torch.float

'set input sequence'
par.N = 50
par.nn = 4
par.Dt = 2

'set noise sources'
par.noise = 0
par.upload_data = 0
par.freq_noise = 0
par.freq = 10
par.jitter_noise = 0
par.jitter = 2

'network model'
par.is_rec = 1
par.w0_rec = -0.07
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

par.rep = 1

'set total length of simulation'
par.T = int(2*(par.Dt*par.N + par.jitter)/(par.dt))


#%%

par.N = 10
par.tau_m = 10.
par.v_th = 2.

'upload stuff'

par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_PyTorch/'+\
		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                par.freq_noise,par.freq)
        

timing = np.load(par.save_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,par.rep),allow_pickle=True)

w = np.load(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec))
v = np.load(par.save_dir+'v_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec),allow_pickle=True)
spk = np.load(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec),allow_pickle=True).tolist()



#%%
par.init_mean = 0.02

par.N = 10
par.nn = 4

par.tau_m = 10.
par.v_th = 2.

par.w0_rec = 0.0


'upload stuff'

par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_PyTorch/'+\
		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                par.freq_noise,par.freq)
        
timing,w,spk, = [], [], []
for rep in range(10):
    print(rep)
        
    timing.append(np.load(par.save_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}.npy'.format(
                                par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,rep),allow_pickle=True))
    
    w.append(np.load(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep)))
#    v.append(np.load(par.save_dir+'v_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
#                                par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True))
    spk.append(np.load(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist())

#%%

selectivity = np.zeros((10,par.batch,par.nn,par.epochs))
output_time = np.zeros((10,par.batch,par.nn,par.epochs))

for rep in range(10):
    for e in range(par.epochs):
        for n in range(par.nn):
            for b in range(par.batch):
                if spk[rep][e][n][b] != []: 
                    selectivity[rep,b,n,e] = 1
                    output_time[rep,b,n,e] = spk[rep][e][n][b][-1]
            
#%%
rep = 9
for b in range(par.batch):
    plt.subplot(par.batch,1,b+1)
    plt.imshow(selectivity[rep,b,:],aspect='auto')

#%%
rep = 2
for b in range(par.batch):
    plt.subplot(par.batch,1,b+1)
    plt.imshow(output_time[rep,b,:],aspect='auto')
    plt.colorbar()
#%%


plt.imshow(w[:,0,:].T,aspect='auto');plt.colorbar()

for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')




