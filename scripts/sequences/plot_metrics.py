"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"plot_metrics.py"
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

N = 50

loaddir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences_N_{}/'.format(N)
savedir = '/mnt/gs/home/saponatim/'


loss_mean = np.load(loaddir+'loss_mean_N_{}.npy'.format(N))
loss_std = np.load(loaddir+'loss_std_N_{}.npy'.format(N))
fig = plt.figure(figsize=(4,6), dpi=300)    
plt.plot(loss_mean/loss_mean[0],color='purple',linewidth=2)    
plt.fill_between(range(len(loss_mean)),loss_mean/loss_mean[0]-loss_std/loss_std[0],
                 loss_mean/loss_mean[0]+loss_std/loss_std[0],'purple',alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.ylabel(r'$\mathcal{L}_{norm}$')
plt.xlim(0,2000)
plt.ylim(0,1.2)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.savefig(savedir+'loss_N_{}.png'.format(N), format='png', dpi=300)
plt.savefig(savedir+'loss_N_{}.pdf'.format(N), format='pdf', dpi=300)
plt.close('all')

v_mean = np.load(loaddir+'v_mean_N_{}.npy'.format(N))
v_std = np.load(loaddir+'v_std_N_{}.npy'.format(N))
fig = plt.figure(figsize=(4,6), dpi=300)    
plt.plot(v_mean/v_mean[0],color='navy',linewidth=2)    
plt.fill_between(range(len(v_mean)),v_mean/v_mean[0]-v_std/v_std[0],
                 v_mean/v_mean[0]+v_std/v_std[0],color='navy',alpha=.3)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.ylabel(r'$\mathcal{L}_{norm}$')
plt.xlim(0,2000)
plt.ylim(0,1.2)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.savefig(savedir+'v_N_{}.png'.format(N), format='png', dpi=300)
plt.savefig(savedir+'v_N_{}.pdf'.format(N), format='pdf', dpi=300)
plt.close('all')

#fr_mean = np.load(loaddir+'fr_mean_N_{}.npy'.format(N))
#fr_std = np.load(loaddir+'fr_std_N_{}.npy'.format(N))
#fig,(ax1,ax2) = plt.subplots(2,1, sharex=True,figsize=(6,4))
#ax1.plot(np.arange(0,len(fr_mean),1),fr_mean,color='purple') 
#ax2.plot(np.arange(0,len(fr_mean),1),fr_mean,color='purple')
#ax1.fill_between(np.arange(0,len(fr_mean),1),fr_mean-fr_std,
#                 fr_mean+fr_std,color='purple',alpha=.3) 
#ax2.fill_between(np.arange(0,len(fr_mean),1),fr_mean-fr_std,
#                 fr_mean+fr_std,color='purple',alpha=.3) 
#fig.tight_layout(rect=[0, 0.01, 1, 0.97])
#ax1.set_ylim(np.max(fr_mean)-1,np.max(fr_mean)+1) 
#ax2.set_ylim(0,np.max(fr_mean[1:])) 
#ax1.spines['bottom'].set_visible(False)
#ax2.spines['top'].set_visible(False)
#ax1.xaxis.tick_top()
#ax1.tick_params(labeltop=False) 
#ax2.xaxis.tick_bottom()
#fig.subplots_adjust(hspace=0.05)
#ax2.set_xlabel(r'epochs')
#ax2.set_ylabel(r'firing rate [Hz]')
#ax2.axhline(y=fr_mean(axis=0).mean(), color='grey',linestyle='dashed',linewidth=1.5)
#ax2.set_xlim(0,len(fr_mean))
#plt.savefig(savedir+'output_fr_N_{}.png'.format(N), format='png', dpi=300)
#plt.savefig(savedir+'output_fr_N_{}.pdf'.format(N), format='pdf', dpi=300)
#plt.close('all')