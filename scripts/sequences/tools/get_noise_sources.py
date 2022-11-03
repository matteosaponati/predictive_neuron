import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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
N_dist = 100
jitter = 2.
freq = 10.

tau_m = 10.
v_th = 1.4
eta = 5e-4
init_mean = .01 
rep = 1

'--------------------------------------------------'
'without any noise'

noise = 0
jitter_noise = 0
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