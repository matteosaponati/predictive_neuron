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
jitter = 2.
freq = 10.

noise = 0
jitter_noise = 0
freq_noise = 0
onset = 0

tau_m = 10.
v_th = 1.4
eta = 5e-4
init_mean = 0.04
rep = 1

savedir = '/mnt/hpc/departmentN4/matteo/predictive_neuron/figures/suppfig_sequence_capacity/plots/'

'--------------------------------------------------'

tau_m_sweep = [5.,10.,15.,20.,25.,30.]
v_th_sweep = [.5,1.,1.5,2.,2.5]

'--------------------------------------------------'
N_sub = 5
batch = 20

N_subseq = [np.arange(k,k+N_sub) for k in np.arange(0,batch*N_sub+N_sub,N_sub)]

alpha = np.zeros((len(tau_m_sweep),len(v_th_sweep)))   

for tm in range(len(tau_m_sweep)):
    for th in range(len(v_th_sweep)):    
        
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/multisequences/'+\
    		'Dt_{}_N_sub_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_sub,batch,noise,jitter_noise,jitter,freq_noise,freq,onset)
            
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep))
        
        for k in range(batch):
            
            if w[-1,N_subseq[k][0]:N_subseq[k][-1]].argmax() == 0 \
            and spk[-1][k] != [] \
            and spk[-1][k][-1] < Dt*N_sub: 
                alpha[tm,th] +=1/batch

np.save(savedir+'capacity_taum_vth_N_sub_{}_batch_{}'.format(N_sub,batch),alpha)

fig = plt.figure(figsize=(7,6), dpi=300)
plt.imshow(np.flipud(alpha),cmap='Purples',aspect='auto')
plt.colorbar()
plt.yticks([0,len(tau_m_sweep)],[tau_m_sweep[-2],tau_m_sweep[0]])
plt.xticks([0,len(v_th_sweep)],[v_th_sweep[0],v_th_sweep[-2]])
plt.xlabel(r'$v_{th}$')
plt.ylabel(r'$\tau_{m}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'capacity_N_sub_{}_batch_{}.png'.format(N_sub,batch),format='png', dpi=300)
plt.savefig(savedir+'capacity_N_sub_{}_batch_{}.pdf'.format(N_sub,batch),format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
N_sub = 10
batch = 10

N_subseq = [np.arange(k,k+N_sub) for k in np.arange(0,batch*N_sub+N_sub,N_sub)]

alpha = np.zeros((len(tau_m_sweep),len(v_th_sweep)))   

for tm in range(len(tau_m_sweep)):
    for th in range(len(v_th_sweep)):    
        
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/multisequences/'+\
    		'Dt_{}_N_sub_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_sub,batch,noise,jitter_noise,jitter,freq_noise,freq,onset)
            
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep))
        
        for k in range(batch):
            
            if w[-1,N_subseq[k][0]:N_subseq[k][-1]].argmax() == 0 \
            and spk[-1][k] != [] \
            and spk[-1][k][-1] < Dt*N_sub: 
                alpha[tm,th] +=1/batch

np.save(savedir+'capacity_taum_vth_N_sub_{}_batch_{}'.format(N_sub,batch),alpha)

fig = plt.figure(figsize=(7,6), dpi=300)
plt.imshow(np.flipud(alpha),cmap='Purples',aspect='auto')
plt.colorbar()
plt.yticks([0,len(tau_m_sweep)],[tau_m_sweep[-2],tau_m_sweep[0]])
plt.xticks([0,len(v_th_sweep)],[v_th_sweep[0],v_th_sweep[-2]])
plt.xlabel(r'$v_{th}$')
plt.ylabel(r'$\tau_{m}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'capacity_N_sub_{}_batch_{}.png'.format(N_sub,batch),format='png', dpi=300)
plt.savefig(savedir+'capacity_N_sub_{}_batch_{}.pdf'.format(N_sub,batch),format='pdf', dpi=300)
plt.close('all')

'--------------------------------------------------'
N_sub = 20
batch = 5

N_subseq = [np.arange(k,k+N_sub) for k in np.arange(0,batch*N_sub+N_sub,N_sub)]

alpha = np.zeros((len(tau_m_sweep),len(v_th_sweep)))   

for tm in range(len(tau_m_sweep)):
    for th in range(len(v_th_sweep)):    
        
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/multisequences/'+\
    		'Dt_{}_N_sub_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_sub,batch,noise,jitter_noise,jitter,freq_noise,freq,onset)
            
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep))
        
        for k in range(batch):
            
            if w[-1,N_subseq[k][0]:N_subseq[k][-1]].argmax() == 0 \
            and spk[-1][k] != [] \
            and spk[-1][k][-1] < Dt*N_sub: 
                alpha[tm,th] +=1/batch

np.save(savedir+'capacity_taum_vth_N_sub_{}_batch_{}'.format(N_sub,batch),alpha)

fig = plt.figure(figsize=(7,6), dpi=300)
plt.imshow(np.flipud(alpha),cmap='Purples',aspect='auto')
plt.colorbar()
plt.yticks([0,len(tau_m_sweep)],[tau_m_sweep[-2],tau_m_sweep[0]])
plt.xticks([0,len(v_th_sweep)],[v_th_sweep[0],v_th_sweep[-2]])
plt.xlabel(r'$v_{th}$')
plt.ylabel(r'$\tau_{m}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'capacity_N_sub_{}_batch_{}.png'.format(N_sub,batch),format='png', dpi=300)
plt.savefig(savedir+'capacity_N_sub_{}_batch_{}.pdf'.format(N_sub,batch),format='pdf', dpi=300)
plt.close('all')