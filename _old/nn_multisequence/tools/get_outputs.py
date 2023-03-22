import numpy as np
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

par = types.SimpleNamespace()

'training algorithm'
par.bound = 'None'
par.init = 'uniform'
par.init_mean = 0.02
par.init_a, par.init_b = 0, .03
par.epochs = 2000
par.batch = 2

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
'-------------------------------------------------'

'1 SEQUENCE CASE'
par.batch = 1

par.init_mean = 0.03

par.N = 50
par.nn = 4

par.tau_m = 10.
par.v_th = 3.
par.eta = 4e-5

par.w0_rec = 0.0

rep = 0

'upload stuff'
par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                par.freq_noise,par.freq)
        
'no recurrent interactions'
par.w0_rec = 0.0

w = np.load(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep))
spk = np.load(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist()

'check weights and spiking activity'
selectivity = np.zeros((par.nn,par.epochs))
output_time = np.zeros((par.nn,par.epochs))

for e in range(par.epochs):
    for n in range(par.nn):
            if spk[e][n][0] != []: 
                selectivity[n,e] = 1
                output_time[n,e] = spk[e][n][0][-1]
                
plt.imshow(w[:,2,:].T,aspect='auto')

'----------------------------------------------------'
'recurrent interactions'
par.w0_rec = -0.04

w = np.load(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep))
spk = np.load(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist()

'distance between weights'
w_dist =  np.zeros((par.nn,par.nn,par.N))
for n in range(par.nn):
    w_dist[n,:,:] = w[0,:] - w[0,n,:] 

'check weights and spiking activity'
selectivity = np.zeros((par.nn,par.epochs))
output_time = np.zeros((par.nn,par.epochs))

for e in range(par.epochs):
    for n in range(par.nn):
            if spk[e][n][0] != []: 
                selectivity[n,e] = 1
                output_time[n,e] = spk[e][n][0][-1]
                
n = 0
plt.imshow(w[:,n,:].T,aspect='auto')

plt.imshow(output_time,aspect='auto');plt.colorbar()

#%%
'check weights and spiking activity'

nn = [4,8,10,15,20]
N = [50,100]
tau_m = [10.,20.,30.]
v_th = [1.,2.,3.]
w0_rec = [-.01,-.02,-.03,-.04,-.05,-.06]


selectivity_fin = np.zeros((len(nn),len(N),len(tau_m),len(v_th),len(w0_rec)))
time_fin  = np.zeros((len(nn),len(N),len(tau_m),len(v_th),len(w0_rec)))

for nnidx in range(len(nn)):
    for Nidx in range(len(N)):
        for tauidx in range(len(tau_m)):
            for vthidx in range(len(v_th)):
                for w0idx in range(len(w0_rec)):
                    
                    'upload stuff'
                    par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
                    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                                    par.Dt,nn[nnidx],N[Nidx],par.batch,par.noise,par.jitter_noise,par.jitter,
                                    par.freq_noise,par.freq)
                    
                    w = np.load(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                            tau_m[tauidx],v_th[vthidx],par.eta,par.init_mean,w0_rec[w0idx],rep))
                    spk = np.load(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                            tau_m[tauidx],v_th[vthidx],par.eta,par.init_mean,w0_rec[w0idx],rep),allow_pickle=True).tolist()
                        
                    selectivity = np.zeros((nn[nnidx],par.epochs))
                    output_time = np.zeros((nn[nnidx],par.epochs))
                    
                    for e in range(par.epochs):
                        for n in range(nn[nnidx]):
                                if spk[e][n][0] != []: 
                                    selectivity[n,e] = 1
                                    output_time[n,e] = spk[e][n][0][-1]
                                    
                    print('{} {} {} {} {} '.format(nn[nnidx],N[Nidx],tau_m[tauidx],v_th[vthidx],w0_rec[w0idx]))
                    print(selectivity[:,-1])
                    print(output_time[:,-1])

'-------------------------------------------------'

#%%
'2 SEQUENCES CASE'

par.batch = 2
par.init_mean = 0.03

par.N = 100
par.nn = 4

par.tau_m = 10.
par.v_th = 3.
par.eta = 5e-5

par.w0_rec = 0.0

rep_tot = 15

'upload stuff'
par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                par.freq_noise,par.freq)

w,spk,timing = [],[],[]
selectivity = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
output_time = np.zeros((rep_tot,par.batch,par.nn,par.epochs))


for rep in range(rep_tot):
    w.append(np.load(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep)))
    spk.append(np.load(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist())
    timing.append(np.load(par.save_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}.npy'.format(
                                par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,rep),allow_pickle=True).tolist())


    for b in range(par.batch):
        for e in range(par.epochs):
            for n in range(par.nn):
                    if spk[rep][e][n][b] != []: 
                        selectivity[rep,b,n,e] = 1
                        if spk[rep][e][n][0] != []: output_time[rep,b,n,e] = spk[rep][e][n][0][-1]

#%%
import os
                        
def plot_selectivity(par):           

    
    'set dirs'
    
    par.savedir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy_plots/'+\
    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                        par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                        par.freq_noise,par.freq)
    if not os.path.exists(par.savedir): os.makedirs(par.savedir) 

    par.load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                    par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq)
    
    'set outputs'
    w,spk,timing = [],[],[]
    selectivity = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    output_time = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    
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
                            
                            'get selectivity'
                            selectivity[rep,b,n,e] = 1
                            'get output times'
                            output_time[rep,b,n,e] = spk[rep][e][n][b][-1]     

    'plot selectivity final'
    fig = plt.figure(figsize=(6,8), dpi=300)
    plt.subplot(2,1,1)
    batch=0
    plt.imshow(selectivity[:,batch,:,-1].T,aspect='auto')
    plt.ylabel('neurons')
    plt.subplot(2,1,2)
    batch=1
    plt.imshow(selectivity[:,batch,:,-1].T,aspect='auto') 
    plt.ylabel('neurons')
    plt.xlabel('repetitions')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(par.savedir+'selectivity_final_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}.png'.format(
                                    par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean), format='png', dpi=300)
    plt.close('all')
    
    'plot selectivity final'
    fig = plt.figure(figsize=(6,8), dpi=300)
    plt.subplot(2,1,1)
    batch=0
    plt.imshow(output_time[:,batch,:,-1].T,aspect='auto')
    plt.colorbar()
    plt.ylabel('neurons')
    plt.subplot(2,1,2)
    batch=1
    plt.imshow(output_time[:,batch,:,-1].T,aspect='auto') 
    plt.colorbar()
    plt.ylabel('neurons')
    plt.xlabel('repetitions')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(par.savedir+'output_time_final_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}.png'.format(
                                    par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean), format='png', dpi=300)
    plt.close('all')
    
'sweep'
nn = [10]
N = [100]
tau_m = [10.,20.]
v_th = [2.,3.]
eta = [5e-5,1e-5]
w0_rec = [0.0,-.01,-.02,-.03,-.04,-.05,-.06]

par.init_mean = 0.03
rep_tot = 15

for nnidx in nn:
    
    print(nnidx)
    
    for Nidx in N:
        for tauidx in tau_m:
            for vthidx in v_th:
                for etaidx in eta:
                    for w0idx in w0_rec:    
                        
                        par.nn = nnidx
                        par.N = Nidx
                        par.tau_m = tauidx
                        par.v_th = vthidx
                        par.eta = etaidx
                        par.w0_rec = w0idx
                        
                        plot_selectivity(par)
                        
                        
#%%

                        
def compute_selectivity(par):           

#    'set dirs'
#    par.savedir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy_plots/'+\
#    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
#                        par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
#                        par.freq_noise,par.freq)
#    if not os.path.exists(par.savedir): os.makedirs(par.savedir) 

    par.load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
    		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                    par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq)
    
    'set outputs'
    w,spk,timing = [],[],[]
    selectivity = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    output_time = np.zeros((rep_tot,par.batch,par.nn,par.epochs))
    
    for rep in range(rep_tot):
        w.append(np.load(par.load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep)))
        spk.append(np.load(par.load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}.npy'.format(
                                    par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,rep),allow_pickle=True).tolist())
#        timing.append(np.load(par.load_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}.npy'.format(
#                                    par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,rep),allow_pickle=True).tolist())
        
        for b in range(par.batch):
            for e in range(par.epochs):
                for n in range(par.nn):
                        if spk[rep][e][n][b] != []: 
                            
                            'get selectivity'
                            selectivity[rep,b,n,e] = 1
                            'get output times'
                            output_time[rep,b,n,e] = spk[rep][e][n][b][-1]     
                            
    return selectivity, output_time, w


'sweep'
nn = [4]
N = [50]
tau_m = [20.]
v_th = [2.]
eta = [1e-5]
w0_rec = [0.0,-.01,-.02,-.03,-.04,-.05,-.06]

par.init_mean = 0.03
rep_tot = 15

sel_tot, output_tot, w_tot = [], [], []

for nnidx in nn:
    
    for Nidx in N:
        for tauidx in tau_m:
            for vthidx in v_th:
                for etaidx in eta:
                    for w0idx in w0_rec:    
                        
                        par.nn = nnidx
                        par.N = Nidx
                        par.tau_m = tauidx
                        par.v_th = vthidx
                        par.eta = etaidx
                        par.w0_rec = w0idx
                        
                        selectivity, output_time, w = compute_selectivity(par)
                        
                        sel_tot.append(selectivity)
                        output_tot.append(output_time)
                        w_tot.append(w)

idx = 4
                        
selective, nonselective, dead  = np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs)), np.zeros((rep_tot,par.epochs))
output_times = np.zeros((rep_tot,par.batch,par.epochs))

for e in range(par.epochs):
    
        for nnidx in range(nn[0]):
            for rep in range(rep_tot):
    
                if sel_tot[idx][rep,:,nnidx,e].sum() == 1: selective[rep,e] +=1
                if sel_tot[idx][rep,:,nnidx,e].sum() == 2: nonselective[rep,e] += 1
                if sel_tot[idx][rep,:,nnidx,e].sum() == 0: dead[rep,e] += 1

for e in range(par.epochs):
    for b in range(par.batch):
#        for nnidx in range(nn[0]):
            for rep in range(rep_tot):
                    
                output_times[rep,b,e] = output_tot[idx][rep,b,:,e].mean()

robustness = np.zeros((rep_tot,par.batch,par.epochs))      
robustness_idx = np.zeros((rep_tot,par.batch,nn[0]))      
for e in range(1,par.epochs):
    for b in range(par.batch):
            for rep in range(rep_tot):
                
                for nnidx in range(nn[0]):
                    if sel_tot[idx][rep,b,nnidx,e] == sel_tot[idx][rep,b,nnidx,e-1]:
                        robustness[rep,b,e] += 1

for b in range(par.batch):
    for rep in range(rep_tot):
        for nnidx in range(nn[0]):
                robustness_idx[rep,b,nnidx] = sel_tot[idx][rep,b,nnidx,:].mean()

plt.plot(selective.mean(axis=0)/nn[0],'r')
plt.plot(nonselective.mean(axis=0)/nn[0],'b')
plt.plot(dead.mean(axis=0)/nn[0],'k')

plt.plot(output_times[:,0].mean(axis=0),'r')
plt.plot(output_times[:,1].mean(axis=0),'b')

plt.plot((robustness[:,0,:]/nn[0]).mean(axis=0),'r')
plt.plot((robustness[:,1,:]/nn[0]).mean(axis=0),'b')


#%%            
    
                        

idx = 4
plt.subplot(2,1,1)
batch=0
plt.imshow(sel_tot[idx][6,batch,:,:],aspect='auto')
plt.subplot(2,1,2)
batch=1
plt.imshow(sel_tot[idx][6,batch,:,:],aspect='auto') 

#%%



'distance between inputs'
## compute how likely two sequence are very different if you shuffle (compute distance between first inputs in the weight space)



spk_times = np.zeros((500,par.batch,par.N))
for k in range(500):
    for b in range(par.batch):
        times = (np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt).astype(int)
        np.random.shuffle(times)
        spk_times[k,b,:] = times
#    timing = [[] for n in range(par.nn)]
#    for n in range(par.nn):
#        for b in range(par.batch): timing[n].append(spk_times[b])

seq_dist, seq_dist_order = np.zeros((500,par.N)), np.zeros((500,par.N))
for k in range(500):
    seq_dist[k,:] = spk_times[k,0,:]-spk_times[k,1,:]
    seq_dist_order[k,:] = spk_times[k,0,:][np.argsort(spk_times[k,0,:])] - spk_times[k,1,:][np.argsort(spk_times[k,0,:])]


'distance between weights'
w_dist =  np.zeros((par.nn,par.nn,par.N))
for n in range(par.nn):
    w_dist[n,:,:] = w[0,:] - w[0,n,:] 
    
    
    







'check weights and spiking activity'
selectivity = np.zeros((par.nn,par.epochs))
output_time = np.zeros((par.nn,par.epochs))

for e in range(par.epochs):
    for n in range(par.nn):
            if spk[e][n][0] != []: 
                selectivity[n,e] = 1
                output_time[n,e] = spk[e][n][0][-1]
                
plt.imshow(w[:,0,:].T,aspect='auto')
plt.plot(output_time.T)




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




