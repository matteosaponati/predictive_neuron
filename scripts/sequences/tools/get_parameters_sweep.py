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
N_dist = 0
jitter = 2.
freq = 10.

noise = 0
jitter_noise = 0
freq_noise = 0
onset = 0

tau_m = 10.
v_th = 1.4
eta = 5e-4
init_mean = .1
rep = 1

savedir = '/mnt/hpc/departmentN4/matteo/predictive_neuron/figures/suppfig_sequence_parameters/plots/'

'--------------------------------------------------'
'tau_m and v_th'

tau_m_sweep = [x for x in np.arange(1.,30.,1)]
v_th_sweep = [x/10 for x in range(1,30,1)]

matrix = np.zeros((len(tau_m_sweep),len(v_th_sweep)))
mask = np.zeros((len(tau_m_sweep),len(v_th_sweep)))    

for tm in range(len(tau_m_sweep)):
    for th in range(len(v_th_sweep)):    
        
        load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)
            
        spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep),allow_pickle=True).tolist()
        w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                                    tau_m_sweep[tm],v_th_sweep[th],eta,init_mean,rep))
        
        if w[-1,:].argmax() == 0 and spk[-1] != [] and spk[-1][-1] < 20 and len(spk[-1])<100: mask[tm,th] = 1        
        if spk[-1] != []: matrix[tm,th] = len(spk[-1])


np.save(savedir+'parspace_taum_vth',matrix)
np.save(savedir+'mask_taum_vth',mask)

fig = plt.figure(figsize=(7,6), dpi=300)
norm = colors.BoundaryNorm(boundaries=np.linspace(0,1000,11),ncolors=256)
plt.imshow(np.flipud(matrix),cmap='Purples',norm=norm,aspect='auto')
plt.colorbar()
plt.contour(np.flipud(mask),10,colors='black',linewidths=.1)
plt.yticks(np.arange(len(tau_m_sweep))[::4],tau_m_sweep[::-1][::4])
plt.xticks(np.arange(len(v_th_sweep))[::4],v_th_sweep[::4])
plt.xlabel(r'$v_{th}$')
plt.ylabel(r'$\tau_{m}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'parameter_space_taum_vth.png',format='png', dpi=300)
plt.savefig(savedir+'parameter_space_taum_vth.pdf',format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(7,6), dpi=300)
plt.imshow(np.flipud(mask),cmap='coolwarm',aspect='auto')
plt.yticks(np.arange(len(tau_m_sweep))[::5],tau_m_sweep[::5][::-1])
plt.xticks(np.arange(len(v_th_sweep))[::10],v_th_sweep[::10])
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlim(0,11)
plt.colorbar()
plt.xlabel(r'$v_{th}$')
plt.ylabel(r'$\tau_{m}$')
plt.savefig(savedir+'mask_taum_vth.png',format='png', dpi=300)
plt.savefig(savedir+'mask_taum_vth.pdf',format='pdf', dpi=300)
plt.close('all')

