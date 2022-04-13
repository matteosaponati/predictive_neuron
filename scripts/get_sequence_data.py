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

dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'
savedir = '/gs/home/saponatim/'

loss = np.load(dir+'loss.npy')
w = np.load(dir+'w.npy')
spk = np.load(dir+'spk.npy',allow_pickle=True)

#%%

fig = plt.figure(figsize=(7,6), dpi=300)
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.pcolormesh(w.T,cmap='coolwarm')
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'w_conv.png', format='png', dpi=300)
plt.close('all')
#%%
fig = plt.figure(figsize=(6,5), dpi=300)
for k,j in zip(spk,range(len(spk))):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'spk.png',format='png', dpi=300)
plt.close('all')

    
    
#%%

dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/fig2/'
savedir = '/gs/home/saponatim/'
N = 10
rep = 1

loss = np.load(dir+'loss_N_{}_rep_{}.npy'.format(N,rep))
w = np.load(dir+'w_N_{}_rep_{}.npy'.format(N,rep))
spk = np.load(dir+'spk_N_{}_rep_{}.npy'.format(N,rep),allow_pickle=True)

#%%

fig = plt.figure(figsize=(7,6), dpi=300)
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.pcolormesh(w.T,cmap='coolwarm')
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'w_conv.png', format='png', dpi=300)
plt.close('all')


#%%
fig = plt.figure(figsize=(6,5), dpi=300)
for k,j in zip(spk,range(len(spk))):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'spk.png',format='png', dpi=300)
plt.close('all')

    

