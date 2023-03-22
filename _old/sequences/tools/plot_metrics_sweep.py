"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"get_loss_sweep.py"
compute how sequence prediction changes in parameter space

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""
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

loaddir = '/Users/saponatim/Desktop/metrics/'
savedir = '/Users/saponatim/Desktop/'

N_sweep = [10,50,100,500]
c = ['firebrick','lightblue','purple','navy']
loss_list = []

fig = plt.figure(figsize=(4,6), dpi=300)
for k in range(len(N_sweep)):
    loss_mean = np.load(loaddir+'loss_mean_N_{}.npy'.format(N_sweep[k]))
    loss_std = np.load(loaddir+'loss_std_N_{}.npy'.format(N_sweep[k]))
    plt.plot(loss_mean/loss_mean[0],color=c[k],linewidth=2,label='N {}'.format(N_sweep[k]))    
    plt.fill_between(range(len(loss_mean)),loss_mean/loss_mean[0]-loss_std/loss_std[0],loss_mean/loss_mean[0]+loss_std/loss_std[0],color=c[k],alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.ylabel(r'$\mathcal{L}_{norm}$')
plt.xlim(0,2000)
plt.ylim(0,1.2)
plt.legend()
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.savefig(savedir+'loss_sweep.png', format='png', dpi=300)
plt.savefig(savedir+'loss_sweep.pdf', format='pdf', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(4,6), dpi=300)
for k in range(len(N_sweep)):
    v_mean = np.load(loaddir+'v_mean_N_{}.npy'.format(N_sweep[k]))
    v_std = np.load(loaddir+'v_std_N_{}.npy'.format(N_sweep[k]))
    plt.plot(v_mean/v_mean[0],color=c[k],linewidth=2,label='N {}'.format(N_sweep[k]))    
    plt.fill_between(range(len(v_mean)),v_mean/v_mean[0]-v_std/v_std[0],
                     v_mean/v_mean[0]+v_std/v_std[0],color=c[k],alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.ylabel(r'$\langle v \rangle$')
plt.xlim(0,2000)
plt.ylim(0,1.2)
plt.legend()
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.savefig(savedir+'v_sweep.png', format='png', dpi=300)
plt.savefig(savedir+'v_sweep.pdf', format='pdf', dpi=300)
plt.close('all')