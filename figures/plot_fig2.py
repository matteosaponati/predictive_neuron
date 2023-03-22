import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

from utils.funs import get_dir_results, get_continuous_cmap

'-----------------------------------------'

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'sequence'
par.package = 'NumPy'

par.bound = 'soft'
par.eta = 5e-4
par.batch = 1
par.epochs = 1000
    
par.init = 'fixed'
par.init_mean = .1
    
par.sequence = 'deterministic'
par.Dt = 2
par.N_seq = 100
par.N_dist = 100
par.N = par.N_seq+par.N_dist

par.freq = 10.
par.jitter = 2.
par.onset = 1

par.dt = .05
par.tau_m = 10.
par.v_th = 1.4
par.tau_x = 2.

par.T = int(2*(par.Dt*par.N_seq + par.jitter)/(par.dt))

par.dir_output = '../_results/'

'-----------------------------------------'

from scripts.main_sequence import main

path = get_dir_results(par)

z = np.load(path+'z.npy')
fr = np.load(path+'fr.npy')*1000 # convert in Hz
w = np.load(path+'w.npy')

#onsets = np.load(path_data+'onsets.npy')

fig = plt.figure(figsize=(7,6), dpi=300)    

plt.imshow(np.flipud(w.T)/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))

fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel('inputs')
plt.xlabel('epochs')
plt.savefig('plots/fig2_c.pdf',format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(4,6), dpi=300)

plt.axhline(y=fr.mean(), color='black',linestyle='dashed',linewidth=1.5)
plt.plot(fr,'purple',linewidth=2)    

fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel('firing rate [Hz]')
plt.savefig('plots/fig2_b_fr.pdf',format='pdf', dpi=300)
plt.close('all')

## ADD spike plots