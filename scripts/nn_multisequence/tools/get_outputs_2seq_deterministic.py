'get outputs for 2 sequences case, deterministic case'

import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

par = types.SimpleNamespace()

'training algorithm'
par.init_mean = 0.03
par.epochs = 1000
par.batch = 2

'set input sequence'
par.N = 50
par.nn = 10
par.Dt = 2

'set noise sources'
par.noise = 0
par.upload_data = 0
par.freq_noise = 0
par.freq = 10
par.jitter_noise = 0
par.jitter = 2

'network model'
par.w0_rec = -0.07
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'----------------------------------------------'
'define function to compute selectivity'
def compute_selectivity(par,rep_tot):           

    par.load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                    par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq)
    
    'set outputs'
    w,spk, timing = [],[], []
    selectivity = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    output_time = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    
    'for each repetition, import weights and output activity and compute metrics'
    for rep in range(rep_tot):
        w.append(np.load(par.load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep)))
        spk.append(np.load(par.load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist())
        timing.append(np.load(par.load_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,rep),allow_pickle=True).tolist())
        
        for b in range(par.batch):
            for e in range(par.epochs):
                for n in range(par.nn):
                        if spk[rep][e][n][b] != []: 
                            
                            'get selectivity and output times'
                            selectivity[rep,b,n,e] = 1
                            output_time[rep,b,n,e] = spk[rep][e][n][b][-1]     
                            
    return selectivity, output_time, w, timing

'----------------------------------------------'
'get outputs on inhibition sweep'
nn = 6
N = 10
tau_m = 20.
v_th = 1.5
eta = 1e-5
w0_rec = [0.0,-.01,-.02,-.03,-.04,-.05,-.06,-.07,-.08]

par.init_mean = 0.01
rep_tot = 10

par.epochs = 2000

sel_tot, output_tot, w_tot, timing_tot = [], [], [], []

'get total for different values of inhibition'
for w0idx in w0_rec:    
    
    par.nn = nn
    par.N = N
    par.tau_m = tau_m
    par.v_th = v_th
    par.eta = eta
    par.w0_rec = w0idx
    
    selectivity, output_time, w, timing = compute_selectivity(par,rep_tot)
    
    sel_tot.append(selectivity)
    output_tot.append(output_time)
    w_tot.append(w)
    timing_tot.append(timing)
    
'set one inhibition value and get metrics'
idx = 6

'-------------'
'synaptic weight matrix'
rep = 1

timing_sort = []
for b in range(par.batch):
    timing_sort.append(np.argsort(timing_tot[idx][rep][0][b]))
    
'ordered for sequence 1'
b = 0
plt.figure(figsize=(6,10))
for n in range(nn):
    plt.subplot(nn,1,n+1)
    plt.imshow(w_tot[idx][rep][:,n,:][:,timing_sort[b]].T,aspect='auto')
'ordered for sequence 2'
b = 1
plt.figure(figsize=(6,10))
for n in range(nn):
    plt.subplot(nn,1,n+1)
    plt.imshow(w_tot[idx][rep][:,n,:][:,timing_sort[b]].T,aspect='auto')
    
'final synaptic weights matrix'
w_final = np.zeros((par.N,par.nn))
for n in range(par.nn):
    w_final[:,n] = w_tot[idx][rep][-1,n,:]

'comparison for different batches'
for b in range(par.batch):
    plt.subplot(1,2,b+1)
    plt.imshow(w_final[timing_sort[b],:],aspect='auto');
    plt.colorbar()
    
'difference in the two batches'
plt.imshow(w_final[timing_sort[0],:]-w_final[timing_sort[1],:],aspect='auto');
plt.colorbar()

'---------------------------'
'METRICS'
                     
selective, nonselective, dead  = np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs))
output_times = np.zeros((rep_tot,par.batch,par.epochs))

'get percentage of selective cells'
for e in range(par.epochs):
    for nnidx in range(nn):
        for rep in range(rep_tot):

            if sel_tot[idx][rep,:,nnidx,e].sum() == 1: selective[rep,e] +=1
            if sel_tot[idx][rep,:,nnidx,e].sum() == 2: nonselective[rep,e] += 1
            if sel_tot[idx][rep,:,nnidx,e].sum() == 0: dead[rep,e] += 1

'get output times of the network'
for e in range(par.epochs):
    for b in range(par.batch):
        for rep in range(rep_tot):
            output_times[rep,b,e] = output_tot[idx][rep,b,:,e].mean()

'get robustness and robustness index'
robustness = np.zeros((rep_tot,par.batch,par.epochs))      
robustness_idx = np.zeros((rep_tot,par.batch,nn))      
for e in range(1,par.epochs):
    for b in range(par.batch):
            for rep in range(rep_tot):
                
                for nnidx in range(nn):
                    if sel_tot[idx][rep,b,nnidx,e] == sel_tot[idx][rep,b,nnidx,e-1]:
                        robustness[rep,b,e] += 1

for b in range(par.batch):
    for rep in range(rep_tot):
        for nnidx in range(nn):
                robustness_idx[rep,b,nnidx] = sel_tot[idx][rep,b,nnidx,:].mean()

'plot metrics'

plt.plot(selective.mean(axis=0)/nn,'r')
plt.plot(nonselective.mean(axis=0)/nn,'b')
plt.plot(dead.mean(axis=0)/nn,'k')

plt.plot(output_times[:,0].mean(axis=0),'r')
plt.plot(output_times[:,1].mean(axis=0),'b')

plt.plot((robustness[:,0,:]/nn).mean(axis=0),'r')
plt.plot((robustness[:,1,:]/nn).mean(axis=0),'b')

#%%
from scipy.spatial import distance

'----------------------------------------------'
'get outputs on inhibition sweep'
nn = 10
N = 100
tau_m = 30.
v_th = 3.
eta = 1e-6
w0_rec = [0.0,-.01,-.02,-.03,-.04,-.05,-.06,-.07,-.08,-.09,-.1,-.11]

par.init_mean = 0.03
rep_tot = 100

par.epochs = 1200

'get total for different values of inhibition'
selective_tot, nonselective_tot, dead_tot = [], [], []
output_tot, hamming_tot = [], []

for w0idx in w0_rec:    
    
    print(w0idx)
    
    'set parameters'
    par.nn = nn
    par.N = N
    par.tau_m = tau_m
    par.v_th = v_th
    par.eta = eta
    par.w0_rec = w0idx
    
    'get data'
    par.load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                    par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq)
    
    'set outputs'
    w,spk, timing = [],[], []
    selectivity = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    output_time = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    
    'for each repetition, import output activity and compute metrics'
    for rep in range(rep_tot):
        
        print(rep)
 
        spk = np.load(par.load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist()
 
        for b in range(par.batch):
            for e in range(par.epochs):
                for n in range(par.nn):
                        if spk[e][n][b] != []: 
                            
                            'get selectivity and output times'
                            selectivity[rep,b,n,e] = 1
                            output_time[rep,b,n,e] = spk[e][n][b][-1]     
        
    selective, nonselective, dead  = np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs))
    output_times = np.zeros((rep_tot,par.batch,par.epochs))
    hamming_dist = np.zeros((rep_tot,par.epochs))
        
    'get percentage of selective cells'
    for e in range(par.epochs):
        for nnidx in range(nn):
            for rep in range(rep_tot):
        
                if selectivity[rep,:,nnidx,e].sum() == 1: selective[rep,e] +=1
                if selectivity[rep,:,nnidx,e].sum() == 2: nonselective[rep,e] += 1
                if selectivity[rep,:,nnidx,e].sum() == 0: dead[rep,e] += 1
                
    'get output times of the network'
    for e in range(par.epochs):
        for b in range(par.batch):
            for rep in range(rep_tot):
                output_times[rep,b,e] = output_time[rep,b,:,e].mean()
    
    for e in range(par.epochs):
        for rep in range(rep_tot):
            hamming_dist[rep,e] = distance.hamming(selectivity[rep,0,:,e],
                                                    selectivity[rep,1,:,e])
    
    'save'    
    selective_tot.append(selective.mean(axis=0))
    nonselective_tot.append(nonselective.mean(axis=0))
    dead_tot.append(dead.mean(axis=0))
    output_tot.append(output_times.mean(axis=0))
    hamming_tot.append(hamming_dist.mean(axis=0))




#%%
idx = 5

plt.plot(selective_tot[idx]/nn,'r')
plt.plot(nonselective_tot[idx]/nn,'b')
plt.plot(dead_tot[idx]/nn,'k')

plt.plot(output_times[:,0].mean(axis=0),'r')
plt.plot(output_times[:,1].mean(axis=0),'b')



selective_ihn, nonselective_ihn, dead_ihn = [], [], []
output_ihn, hamming_ihn = np.zeros((2,len(w0_rec))), []
for idx in range(len(w0_rec)):
    
    selective_ihn.append(selective_tot[idx][-1]/par.nn)
    nonselective_ihn.append(nonselective_tot[idx][-1]/par.nn)
    dead_ihn.append(dead_tot[idx][-1]/par.nn)
    hamming_ihn.append(hamming_tot[idx][-1])
    output_ihn[:,idx] = output_tot[idx][:,-1]
    
    
plt.plot(selective_ihn,'r')
plt.plot(nonselective_ihn,'b')
plt.plot(dead_ihn,'k')

plt.plot(hamming_ihn)

plt.plot(output_ihn[0,:],'r')

plt.plot(output_ihn[1,:],'b')
    
    
    
    























from scipy.spatial import distance

idx = 0

hamming_dist = np.zeros((rep_tot,par.epochs))

for e in range(par.epochs):
    for rep in range(rep_tot):
        hamming_dist[rep,e] = distance.hamming(sel_tot[idx][rep,0,:,e],sel_tot[idx][rep,1,:,e])

plt.plot(hamming_dist.mean(axis=0)*par.nn)

#%%

hamming_fin = np.zeros_like(w0_rec)
for idx in range(len(w0_rec)):
    
    hamming_dist = np.zeros((rep_tot,par.epochs))
    
    for e in range(par.epochs):
        for rep in range(rep_tot):
            hamming_dist[rep,e] = distance.hamming(sel_tot[idx][rep,0,:,e],sel_tot[idx][rep,1,:,e])
            
    hamming_fin[idx] = hamming_dist[:,-1].mean()*par.nn
    
    
    
    
#%%

selective_ihn, nonselective_ihn,dead_ihn = [], [], []
for w0idx in w0_rec:    
    
    par.nn = nn
    par.N = N
    par.tau_m = tau_m
    par.v_th = v_th
    par.eta = eta
    par.w0_rec = w0idx
    
    selectivity, output_time, w, timing = compute_selectivity(par,rep_tot)

    selective, nonselective, dead  = np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs))
    output_times = np.zeros((rep_tot,par.batch,par.epochs))
    
    'get percentage of selective cells'
    for e in range(par.epochs):
        for nnidx in range(nn):
            for rep in range(rep_tot):
    
                if selectivity[rep,:,nnidx,e].sum() == 1: selective[rep,e] +=1
                if selectivity[rep,:,nnidx,e].sum() == 2: nonselective[rep,e] += 1
                if selectivity[rep,:,nnidx,e].sum() == 0: dead[rep,e] += 1
                
    selective_ihn.append((selective[:,-1]/nn).mean())
    nonselective_ihn.append((nonselective[:,-1]/nn).mean())
    dead_ihn.append((dead[:,-1]/nn).mean())

plt.plot(selective_ihn,'r')
plt.plot(nonselective_ihn,'b')
plt.plot(dead_ihn,'k')

#%%

'plot specific repetition'
rep = 7
plt.subplot(2,1,1)
batch = 0
plt.imshow(sel_tot[idx][rep,batch,:,:],aspect='auto')
plt.subplot(2,1,2)
batch = 1
plt.imshow(sel_tot[idx][rep,batch,:,:],aspect='auto') 

rep = 7
idx = -1
nidx = 7
plt.imshow(w_tot[idx][rep][:,nidx,:].T,aspect='auto');plt.colorbar()


