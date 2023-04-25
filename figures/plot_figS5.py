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

'-----------------------------------------'

parser = argparse.ArgumentParser()

par = parser.parse_args()

par.name = 'multisequence'
par.package = 'NumPy'

par.bound = 'none'
par.eta = 3e-7
par.batch = 3
par.epochs = 4000
    
par.init = 'fixed'
par.init_mean = .04
    
par.sequence = 'deterministic'
par.Dt = 2
par.N_seq = 8
par.n_afferents = 3
par.delay = 20
par.N = par.N_seq*par.n_afferents 

par.freq = 5.
par.jitter = 1.
par.onset = 0

par.dt = .05
par.tau_m = 18.
par.v_th = 2.6
par.tau_x = 2.

par.T = int((par.Dt*par.N_seq + par.n_afferents*par.delay + 
                     par.jitter)/(par.dt))
        
par.dir_output = '../_results/'

'-----------------------------------------'

path = get_dir_results(par)

z = np.load(path+'z.npy')
w = np.load(path+'w.npy')

fig = plt.figure(figsize=(8,6), dpi=300)

zPlot = []
for b in range(par.epochs):
     zPlot.append((np.where(z[b,0,0,:])[0]*par.dt).tolist())

for k,j in zip(zPlot,range(par.epochs)):
    plt.scatter([j]*len(k),np.array(k),c='rebeccapurple',s=2)

plt.ylabel('time [ms]')
plt.ylim(0,par.T*par.dt)
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('plots/figS5_b.png',format='png', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel('inputs')
plt.xlabel('epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,par.N-1)
plt.imshow(w.T/par.init_mean,aspect='auto',cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig('plots/figS5_c.pdf', format='pdf', dpi=300)
plt.close('all')