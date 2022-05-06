import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs
import torch.nn.functional as F

'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(neuron.v)              

        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'----------------'
def train(par,x_data):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    
    'set model'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
    'initialization'
    if par.init == 'trunc_gauss':
        neuron.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(neuron.w, mean=par.init_mean, std=.1/np.sqrt(par.N),
                                    a=par.init_a,b=par.init_b)
    if par.init == 'fixed':
        neuron.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
        
    'optimizer'
    optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    
    'allocate outputs'
    loss_out = []
    w = np.zeros((par.epochs,par.N))
    v_out, spk_out = [], []
    
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)
        
        'evaluate loss'
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x_data)
        optimizer.zero_grad()
        E.backward()
        optimizer.step()
        
        'output'
        loss_out.append(E.item())
        w[e,:] = neuron.w.detach().numpy()
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return loss_out, w, v_out, spk_out

#%%
'-------------------'    
par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'architecture'
par.N = 500
par.T = int(100/par.dt)
par.freq_pattern = .01
par.seed = 1992
par.batch = 1
par.epochs = 10000
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .04

par.init = 'trunc_gauss'
par.init_mean = .1
par.init_a = 0.
par.init_b = .2

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/patterns/'

#%%

x_data, density = funs.get_pattern_fixed(par)

"""
IMP:
    - compute density of pattern and density of noise
    - show the effect of the two components on learning the sequence
    - show how this can depend on neuronal parameters
"""

#%%
fig = plt.figure(figsize=(7,4), dpi=300)
plt.plot(np.array(density)/(par.N),linewidth=2,color='navy')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel(r'fr [1/$\tau_m$]')
plt.savefig(par.dir+'pattern_density.png',format='png', dpi=300)
plt.close('all')

#%%

fig = plt.figure(figsize=(7,4), dpi=300)
plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')
plt.xticks(np.arange(par.T)[::2000],np.linspace(0,par.T*par.dt,par.T)[::2000].round(0))
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(par.dir+'pattern_unsorted.png',format='png', dpi=300)
plt.close('all')

x = x_data.clone().detach().numpy()
order = np.zeros(par.N)

#%%

for k in range(par.N):
    if np.nonzero(x[0,:,k])[0] != []:
        order[k] = np.nonzero(x[0,:,k])[0][0]
        
fig = plt.figure(figsize=(7,4), dpi=300)
plt.pcolormesh(x[0,:,np.argsort(order)],cmap='Greys')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xticks(np.arange(par.T)[::500],np.linspace(0,par.T*par.dt,par.T)[::500].round(0))
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(par.dir+'pattern_sorted.png',format='png', dpi=300)

#%%

loss, w, v, spk = train(par,x_data)

#%%
"""
GOOD PLOTS
"""
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

savedir = '/mnt/gs/home/saponatim/'


fig = plt.figure(figsize=(5,6), dpi=300)
plt.pcolormesh(np.linspace(0,par.T*par.dt,par.T),np.arange(par.N),x_data[0,:,:].T,cmap='Greys')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(savedir+'pattern_unsorted.png',format='png', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(5,6), dpi=300)
plt.pcolormesh(np.linspace(0,par.T*par.dt,par.T),np.arange(par.N),x_data[0,:,np.argsort(order)].T,cmap='Greys')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('inputs (sorted)')
plt.savefig(savedir+'pattern_sorted.png',format='png', dpi=300)
plt.close('all')

fig = plt.figure(figsize=(5,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),k,edgecolor='royalblue',facecolor='none',s=7)
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(savedir+'spk.png',format='png', dpi=300)
plt.close('all')




x = x_data.clone().detach().numpy()
order = np.zeros(par.N)

fig = plt.figure(figsize=(5,6), dpi=300)
plt.title(r'$\vec{w}$')
plt.pcolormesh(w[:,np.argsort(order)].T,cmap='RdBu_r',norm=MidpointNormalize(midpoint=0))
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel(r'inputs (sorted)')
plt.savefig(savedir+'w_sorted.png',format='png', dpi=300)

fig = plt.figure(figsize=(5,6), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk.png',format='png', dpi=300)
plt.close('all')

#%%

fig = plt.figure(figsize=(7,11), dpi=300)
plt.subplot(3,1,1)
plt.pcolormesh(x[0,:,np.argsort(order)],cmap='Greys')
plt.xticks(np.arange(par.T)[::500],np.linspace(0,par.T*par.dt,par.T)[::500].astype(int))
#for k in range(len(spk[-1])):
#    plt.axvline(x = spk[-1][k]/par.dt,color='mediumvioletred')
plt.xlabel('time [ms]')
plt.xlim(0,par.T)
plt.ylabel('inputs')
plt.subplot(3,1,2)
plt.plot(np.array(density)/(par.N),linewidth=2,color='navy')
for k in range(len(spk[-1])):
    plt.axvline(x = spk[-1][k],color='mediumvioletred')
plt.xlabel('time [ms]')
plt.xlim(0,int(par.T*par.dt))
plt.ylabel(r'fr [1/$\tau_m$]')
plt.subplot(3,1,3)
plt.pcolormesh(w[:,np.argsort(order)].T,cmap='coolwarm')
plt.colorbar()
plt.xlabel('epochs')
plt.ylabel(r'$\vec{w}$')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk_density.png',format='png', dpi=300)
plt.close('all')

#%%

fig = plt.figure(figsize=(7,4), dpi=300)
plt.plot(w[-1,np.argsort(order)],color='navy')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
#plt.xlabel('epochs')
plt.ylabel(r'$\vec{w}$')
plt.savefig(par.dir+'w_final.png',format='png', dpi=300)

#%%
import seaborn as sns
from scipy import stats

sns.distplot(w[-1,:],fit=stats.lognorm,color='navy')


#%%
fig = plt.figure(figsize=(6,4), dpi=300)
sns.distplot(w[-1,:],fit=stats.lognorm,color='navy')
#plt.xscale('log')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('count')
plt.xlabel(r'w')
plt.savefig(par.dir+'w_hist.png',format='png', dpi=300)


#%%
fig = plt.figure(figsize=(6,5), dpi=300)
for k,j in zip(spk,range(par.epochs)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.savefig(par.dir+'spk.png',format='png', dpi=300)
plt.close('all')

