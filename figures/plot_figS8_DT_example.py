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
from utils.data import get_spike_times

from scipy import stats

'-----------------------------------------'

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'selforg'
par.network_type = 'random'
par.package = 'NumPy'

par.bound = 'none'
par.eta = 1e-6
par.batch = 1
par.epochs = 3000
    
par.init = 'random'
par.init_mean = .06
par.init_rec = .0003
    
par.Dt = 2
par.n_in = 2
par.nn = 8
par.delay = 8
par.n_afferents = 3

par.freq = 5.
par.jitter = 1.

par.dt = .05
par.tau_m = 25.
par.v_th = 3.1
par.tau_x = 2.

par.rep = 1

par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                        par.jitter + 80)/(par.dt))
    
par.N_in = par.n_in*par.nn
par.N = par.N_in+par.nn
    
par.dir_output = '../_results/'

'-----------------------------------------'

path = get_dir_results(par)

time = np.zeros(par.epochs)
z = np.load(path+'z.npy')

for b in range(par.epochs):
        
    if len(np.where(z[b,0,0,:,:])[1]) >0:

        first_spk = np.where(z[b,0,0,:,:])[1].min()*par.dt
        last_spk = np.where(z[b,0,0,:,:])[1].max()*par.dt
        
        time[b] = last_spk-first_spk

fig = plt.figure(figsize=(4,6),dpi=300)
plt.plot(time,color='purple')
plt.xlabel('epochs')
plt.ylabel(r'$\Delta t$ [ms]')
plt.savefig('plots/figS8_DT_example.pdf',format='pdf',dpi=300)

