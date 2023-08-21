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

from utils.funs import get_dir_results

from scipy import stats

'-----------------------------------------'

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'selforg'
par.network_type = 'nearest'
par.package = 'NumPy'

par.bound = 'none'
par.eta = 8e-7
par.batch = 1
par.epochs = 2000
    
par.init = 'fixed'
par.init_mean = .02
par.init_rec = .0003
    
par.Dt = 2
par.n_in = 8
par.nn = 10
par.delay = 4

par.freq = 5.
par.jitter = 1.

par.dt = .05
par.tau_m = 26.
par.v_th = 3.
par.tau_x = 2.

par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
    
par.N = par.n_in+2
    
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
plt.savefig('plots/fig3_DT_example.pdf',format='pdf',dpi=300)

