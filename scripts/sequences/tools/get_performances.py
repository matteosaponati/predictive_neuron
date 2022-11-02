import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import sem
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

'--------------------------------------------------'

Dt = 2
N_seq = 100
N_dist = 0
jitter = 2.
freq = 10.

tau_m = 10.
v_th = 1.4
eta = 5e-4
init_mean = .01 
rep = 1

savedir = '/mnt/hpc/departmentN4/matteo/predictive_neuron/figures/suppfig_sequence_performance/plots/'

'--------------------------------------------------'
'freq'
noise = 1
jitter_noise = 0
freq_noise = 1
onset = 0

freq_sweep = [1.,10.,20.,30.,50.,100.]

error_freq = np.zeros((100,len(freq_sweep)))

for k in range(len(freq_sweep)):
    for rep in range(100):
    
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq_sweep[k],onset)
    
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != []: error_freq[rep,k] = 1
    
'plot'
fig = plt.figure(figsize=(5,6), dpi=300)
plt.errorbar(freq_sweep,1- error_freq.mean(axis=0),sem(error_freq,axis=0),fmt='-o',color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('firing rate [Hz]')
plt.ylabel('error [%]')
plt.ylim(-.02,1.02)
plt.savefig(savedir+'/freq.png', format='png', dpi=300)
plt.savefig(savedir+'/freq.pdf', format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
'jitter'
noise = 1
jitter_noise = 1
freq_noise = 0
onset = 0

jitter_sweep = np.arange(1.,20.,1.)

error_jitter = np.zeros((100,len(jitter_sweep)))

for k in range(len(jitter_sweep)):
    for rep in range(1,100):
    
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_seq,N_dist,noise,jitter_noise,jitter_sweep[k],freq_noise,freq,onset)
    
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != [] and spk[-1][-1] < 20: error_jitter[rep,k] = 1
        
'plot'
fig = plt.figure(figsize=(5,6), dpi=300)
plt.errorbar(jitter_sweep,1- error_jitter.mean(axis=0),sem(error_jitter,axis=0),fmt='-o',color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('jitter [ms]')
plt.ylabel('error [%]')
plt.ylim(-.02,1.02)
plt.savefig(savedir+'/jitter.png', format='png', dpi=300)
plt.savefig(savedir+'/jitter.pdf', format='pdf', dpi=300)
plt.close('all')
    
'--------------------------------------------------'
'distractors'
noise = 1
jitter_noise = 1
freq_noise = 1
onset = 0

dist_sweep = [50,100,150,200,250,300]

error_dist = np.zeros((100,len(dist_sweep)))

for k in range(len(dist_sweep)):
    for rep in range(100):
    
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_seq,dist_sweep[k],noise,jitter_noise,jitter,freq_noise,freq,onset)
    
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != []: error_dist[rep,k] = 1

'plot'
fig = plt.figure(figsize=(5,6), dpi=300)
plt.errorbar(np.array(dist_sweep)/N_seq,1- error_dist.mean(axis=0),sem(error_dist,axis=0),fmt='-o',color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'% distractors [$N_{dist} / N_{seq}$]')
plt.ylabel('error [%]')
plt.ylim(-.02,1.02)
plt.savefig(savedir+'/distractors.png', format='png', dpi=300)
plt.savefig(savedir+'/distractors.pdf', format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
'initial conditions'
noise = 1
jitter_noise = 1
freq_noise = 1
onset = 0
N_dist = 100

ic_sweep = [.001,.005,.01,.05,.1]

error_ic = np.zeros((100,len(ic_sweep)))

for k in range(len(ic_sweep)):
    for rep in range(100):
    
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)
    
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,ic_sweep[k],rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,ic_sweep[k],rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != []: error_ic[rep,k] = 1
        
'plot'
fig = plt.figure(figsize=(5,6), dpi=300)
plt.errorbar(ic_sweep,1- error_ic.mean(axis=0),sem(error_ic,axis=0),fmt='-o',color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$|\vec{w}_0|$ [a.u.]')
plt.ylabel('error [%]')
plt.xscale('log')
plt.ylim(-.02,1.02)
plt.savefig(savedir+'/w0.png', format='png', dpi=300)
plt.savefig(savedir+'/w0.pdf', format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
'sequence length '
noise = 1
jitter_noise = 1
freq_noise = 1
onset = 0
N_dist = 100

Nseq_sweep = [10,50,100,150,200,250,300]

error_seq = np.zeros((100,len(Nseq_sweep)))

for k in range(len(Nseq_sweep)):
    for rep in range(100):
    
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,Nseq_sweep[k],N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)
    
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != []: error_seq[rep,k] = 1
        
'plot'
fig = plt.figure(figsize=(5,6), dpi=300)
plt.errorbar(ic_sweep,1- error_ic.mean(axis=0),sem(error_ic,axis=0),fmt='-o',color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'$|\vec{w}_0|$ [a.u.]')
plt.ylabel('error [%]')
plt.xscale('log')
plt.ylim(-.02,1.02)
plt.savefig(savedir+'/w0.png', format='png', dpi=300)
plt.savefig(savedir+'/w0.pdf', format='pdf', dpi=300)
plt.close('all')
'--------------------------------------------------'
'freq'
noise = 1
jitter_noise = 1
freq_noise = 1
onset = 0
N_dist = 0

size_sweep = [50,100,150,200,250]

error_size = np.zeros((100,len(size_sweep)))

for k in range(len(size_sweep)):
    for rep in range(100):
    
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,size_sweep[k],N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)
    
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m,v_th,eta,init_mean,rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != []: error_size[rep,k] = 1
        
#%%

from scipy.stats import sem
plt.errorbar(ic_sweep,1- error_ic.mean(axis=0),sem(error_ic,axis=0),fmt='*',color='purple')
plt.xscale('log')
plt.ylim(0,1)
        
#%%

w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(w_plot.shape[0])):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,Dt*N_seq+jitter)
plt.xlim(0,w_plot.shape[0])
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/spk.png', format='png', dpi=300)
plt.savefig(load_dir+'/spk.pdf', format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,w_plot.shape[0])
plt.ylim(0,w_plot.shape[1])
plt.imshow(w_plot.T/init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/w.png', format='png', dpi=300)
plt.savefig(load_dir+'/w.pdf', format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
'onset'

noise = 1
jitter_noise = 0
freq_noise = 0
onset = 1

load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                    	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)

spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep))

w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(w_plot.shape[0])):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,Dt*N_seq+jitter)
plt.xlim(0,w_plot.shape[0])
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/spk.png', format='png', dpi=300)
plt.savefig(load_dir+'/spk.pdf', format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,w_plot.shape[0])
plt.ylim(0,w_plot.shape[1])
plt.imshow(w_plot.T/init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/w.png', format='png', dpi=300)
plt.savefig(load_dir+'/w.pdf', format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
'jitter'

noise = 1
jitter_noise = 1
freq_noise = 0
onset = 0

load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                    	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)

spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep))

w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(w_plot.shape[0])):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,Dt*N_seq+jitter)
plt.xlim(0,w_plot.shape[0])
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/spk.png', format='png', dpi=300)
plt.savefig(load_dir+'/spk.pdf', format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,w_plot.shape[0])
plt.ylim(0,w_plot.shape[1])
plt.imshow(w_plot.T/init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/w.png', format='png', dpi=300)
plt.savefig(load_dir+'/w.pdf', format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
'freq'

noise = 1
jitter_noise = 0
freq_noise = 1
onset = 0

load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                    	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)

spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep))

w_plot = np.vstack(w)
fig = plt.figure(figsize=(7,6), dpi=300)
for k,j in zip(spk,range(w_plot.shape[0])):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)
plt.ylabel(r'time [ms]')
plt.ylim(0,Dt*N_seq+jitter)
plt.xlim(0,w_plot.shape[0])
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/spk.png', format='png', dpi=300)
plt.savefig(load_dir+'/spk.pdf', format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,w_plot.shape[0])
plt.ylim(0,w_plot.shape[1])
plt.imshow(w_plot.T/init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(load_dir+'/w.png', format='png', dpi=300)
plt.savefig(load_dir+'/w.pdf', format='pdf', dpi=300)
plt.close('all')		