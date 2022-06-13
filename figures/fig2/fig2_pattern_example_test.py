import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs

import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))


savedir = '/mnt/gs/home/saponatim/'

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
def train(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'set model'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
    x_data, density, fr = funs.get_pattern(par,mu=True)
    
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
        
        x_data, density, fr = funs.get_pattern(par,mu=True)
        
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
    

par = types.SimpleNamespace()

'model parameters'
par.dt = .05
par.eta = 1e-3
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.

'architecture'
par.N = 500
par.T = int(200/par.dt)
par.T_pattern = int(100/par.dt)
par.freq_pattern = .01
par.seed = 1992
par.batch = 1
par.epochs = 3000
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .03

par.init = 'trunc_gauss'
par.init_mean = .03
par.init_a = 0.01
par.init_b = .05

par.freq = .01
par.jitter = 2

par.offset = 'False'
par.fr_noise = 'False'
par.jitter_noise = 'True'

par.mask = torch.rand(par.batch,par.T_pattern,par.N).to(par.device)

#%%
par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/patterns/'

'get order'
x_data, density, fr = funs.get_pattern(par)

plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')

#%%


    
#%%
    
x_data2, density, fr = funs.get_pattern(par)

#%%
plt.pcolormesh(x_data2[0,:,:].T,cmap='Greys')

x = x_data.clone().detach().numpy()
order = np.zeros(par.N)
for k in range(par.N):
    if np.nonzero(x[0,:,k])[0] != []:
        order[k] = np.nonzero(x[0,:,k])[0][0]        
order[order == 0.] = np.max(order)+1
plt.pcolormesh(x[0,:,np.argsort(order)],cmap='Greys')      

#%%

loss, w, v, spk = train(par)

#%%

np.save(par.dir+'w_pattern_',w)
np.save(par.dir+'spk_sequence',spk)
np.save(par.dir+'v_sequence',v)
np.save(par.dir+'loss_sequence',loss)

#%%

savedir = '/mnt/gs/home/saponatim/'

'--------------------'
'example inputs'

'plot background'
fig = plt.figure(figsize=(2,4), dpi=300)
offset=1
for k in range(par.N):
    bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
    plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
    offset += 1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.xlim(0,par.T/2)
plt.ylabel('inputs')
plt.savefig(savedir+'pattern_background.png',format='png', dpi=300)
plt.savefig(savedir+'pattern_background.pdf',format='pdf', dpi=300)
plt.close('all')

'plot pattern'
fig = plt.figure(figsize=(4,4), dpi=300)
offset=1
start = np.random.randint(0,par.T/2)
for k in range(par.N):
    bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
    bg = [i for i in bg if i<(start) or i>(start+par.T_pattern)]
    plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
    pattern = torch.where(par.mask[0,:,k] < par.freq_pattern*par.dt)[0].numpy()
    pattern += start
    plt.eventplot([pattern],lineoffsets = offset,linelengths = 3,linewidths = .5,colors = 'purple')
    offset += 1
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.xlim(0,par.T)
plt.ylabel('inputs')
plt.savefig(savedir+'pattern.png',format='png', dpi=300)
plt.savefig(savedir+'pattern.pdf',format='pdf', dpi=300)
plt.close('all')    

density,fr = funs.get_pattern_density(par,mu=True,offset=start)

'plot density'
fig = plt.figure(figsize=(4,4), dpi=300)
plt.plot(density,linewidth=2,color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel(r'density [spk/$\tau_m$]')
plt.savefig(savedir+'pattern_density.png',format='png', dpi=300)
plt.savefig(savedir+'pattern_density.pdf',format='pdf', dpi=300)
plt.close('all')

'-------'
    
'plot average firing rate'
repetitions = 1000
fr_avg = []

for k in range(repetitions):
    start = np.random.randint(0,par.T/2)
    _, fr = funs.get_pattern_density(par,mu=True,offset=start)
    fr_avg.append(fr)
fr_m = np.mean(np.array(fr_avg),axis=0)
fr_std = np.std(np.array(fr_avg),axis=0)

fig = plt.figure(figsize=(4,4), dpi=300)
plt.plot(fr_m,linewidth=2,color='k')
plt.fill_between(range(200),fr_m + .5*fr_std, fr_m - .5*fr_std,alpha=.5,color='grey')
plt.ylim(0,20)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel(r'firing rate [Hz]')
plt.savefig(savedir+'pattern_fr.png',format='png', dpi=300)
plt.savefig(savedir+'pattern_fr.pdf',format='pdf', dpi=300)
plt.close('all')

hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
fig = plt.figure(figsize=(7,4), dpi=300)       
divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
plt.imshow(np.flip(w[:,np.argsort(order)].T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.ylabel('inputs')
plt.xlabel(r'epochs')
plt.savefig(savedir+'w_spk_pattern.png',format='png', dpi=300)
plt.savefig(savedir+'w_spk_pattern.pdf',format='pdf', dpi=300)
plt.close('all')

spk_eff = []
for k in range(len(spk)):
     spk_eff.append(spk[k][::4])
     
fig = plt.figure(figsize=(7,4), dpi=300)
for k,j in zip(spk_eff,range(par.epochs)):
    plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.ylim(0,200)
plt.ylabel('spk times [ms]')
plt.grid(True,which='both',color='darkgrey',linewidth=.7)
plt.savefig(savedir+'spk_pattern.png',format='png', dpi=300)
plt.savefig(savedir+'spk_pattern.pdf',format='pdf', dpi=300)
plt.close('all')

