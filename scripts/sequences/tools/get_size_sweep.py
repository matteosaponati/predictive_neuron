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

tau_m = 10.
v_th = 1.4
eta = 5e-4
rep = 0

savedir = '/mnt/hpc/departmentN4/matteo/predictive_neuron/figures/suppfig_sequence_parameters/plots/'

'--------------------------------------------------'
'tau_m and v_th'

N_seq = 100
N_dist = 100

noise = 1
jitter_noise = 1
freq_noise = 1
onset = 0
init_mean = .4

load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                        	Dt,N_seq,N_dist,noise,jitter_noise,jitter,freq_noise,freq,onset)
            
spk = np.load(load_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep),allow_pickle=True).tolist()
w = np.load(load_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}.npy'.format(
                            tau_m,v_th,eta,init_mean,rep))