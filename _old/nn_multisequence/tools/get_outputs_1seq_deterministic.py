'get outputs for 1 sequence case, deterministic case'

import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

par = types.SimpleNamespace()

'training algorithm'
par.init_mean = 0.03
par.epochs = 500
par.batch = 1

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
    w,spk,timing = [],[],[]
    selectivity = np.zeros((rep_tot,par.nn,par.epochs))
    output_time = np.zeros((rep_tot,par.nn,par.epochs))
    
    'for each repetition, import weights and output activity and compute metrics'
    for rep in range(rep_tot):
        w.append(np.load(par.load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep)))
        spk.append(np.load(par.load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist())
        timing.append(np.load(par.load_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,rep),allow_pickle=True).tolist())
        
        for e in range(par.epochs):
            for n in range(par.nn):
                    if spk[rep][e][n][0] != []: 
                        
                        'get selectivity and output times'
                        selectivity[rep,n,e] = 1
                        output_time[rep,n,e] = spk[rep][e][n][0][-1]     
                            
    return selectivity, output_time, w, timing

'----------------------------------------------'
'get outputs on inhibition sweep'
nn = 8
N = 50
tau_m = 30.
v_th = 4.
eta = 5e-7
w0_rec = [-.01,-.02,-.03,-.04,-.05,-.06]

par.init_mean = 0.03
rep_tot = 10

par.epochs = 1500

sel_tot, output_tot, w_tot,timing_tot = [], [], [], []

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
idx = 0
                        
selective, dead  = np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs))
output_times = np.zeros((rep_tot,par.epochs))

'get percentage of selective cells'
for e in range(par.epochs):
    for nnidx in range(nn):
        for rep in range(rep_tot):

            if sel_tot[idx][rep,nnidx,e].sum() == 1: selective[rep,e] +=1
            if sel_tot[idx][rep,nnidx,e].sum() == 0: dead[rep,e] += 1

'get output times of the network'
for e in range(par.epochs):
        for rep in range(rep_tot):
            output_times[rep,e] = output_tot[idx][rep,:,e].mean()

'get robustness and robustness index'
robustness = np.zeros((rep_tot,par.epochs))      
robustness_idx = np.zeros((rep_tot,par.nn))      
for e in range(1,par.epochs):
            for rep in range(rep_tot):
                
                for nnidx in range(nn):
                    if sel_tot[idx][rep,nnidx,e] == sel_tot[idx][rep,nnidx,e-1]:
                        robustness[rep,e] += 1

for rep in range(rep_tot):
    for nnidx in range(nn):
            robustness_idx[rep,nnidx] = sel_tot[idx][rep,nnidx,:].mean()
            
'plot metrics'

plt.plot(selective.mean(axis=0)/nn,'r')
plt.plot(dead.mean(axis=0)/nn,'k')

plt.plot(output_times[0,:],'r')

plt.plot(robustness[0,:]/nn,'r')

rep = 0
plt.figure(figsize=(6,10))
for n in range(nn):
    plt.subplot(nn,1,n+1)
    timing_sort = np.argsort(timing_tot[idx][rep][n][0])
    plt.imshow(w_tot[idx][rep][:,n,:][:,timing_sort].T,aspect='auto')
    
'effect of inhibition'
selective_ihn, nonselective_ihn,dead_ihn = [], [], []
for w0idx in w0_rec:    
    
    par.nn = nn
    par.N = N
    par.tau_m = tau_m
    par.v_th = v_th
    par.eta = eta
    par.w0_rec = w0idx
    
    selectivity, output_time, w, _ = compute_selectivity(par,rep_tot)
    
    selective, nonselective, dead  = np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs))
    
    'get percentage of selective cells'
    for e in range(par.epochs):
        for nnidx in range(nn):
            for rep in range(rep_tot):
    
                if selectivity[rep,nnidx,e].sum() == 1: selective[rep,e] +=1
                if selectivity[rep,nnidx,e].sum() == 0: dead[rep,e] += 1
                
    selective_ihn.append((selective[:,-1]/nn).mean())
    nonselective_ihn.append((nonselective[:,-1]/nn).mean())
    dead_ihn.append((dead[:,-1]/nn).mean())

plt.plot(np.abs(w0_rec),selective_ihn,'r')
plt.plot(np.abs(w0_rec),nonselective_ihn,'b')
plt.plot(np.abs(w0_rec),dead_ihn,'k')

