import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models, funs
import torch.nn.functional as F

def get_pattern(par):
        
    prob = par.freq_pattern*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1
    
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float()
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)

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
par.v_th = 7.
par.tau_x = 2.

'architecture'
par.N = 1000
par.T = int(50/par.dt)
par.freq_pattern = .05
par.seed = 1992
par.batch = 1
par.epochs = 1000
par.device = 'cpu'

par.init = 'fixed'
par.w_0 = .03

#par.init = 'trunc_gauss'
#par.init_mean = 1.
#par.init_a = 0.
#par.init_b = 2.

par.dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/patterns/'

#%%

x_data = get_pattern(par)

#%%

fig = plt.figure(figsize=(7,4), dpi=300)
plt.pcolormesh(x_data[0,:,:].T,cmap='Greys')
plt.xticks(np.arange(par.T)[::2000],np.linspace(0,par.T*par.dt,par.T)[::2000].round(0))
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(par.dir+'pattern_unsorted.png',format='png', dpi=300)
plt.close('all')

#%%

loss, w, v, spk = train(par,x_data)

#%%

x = x_data.clone().detach().numpy()
order = np.zeros(par.N)

for k in range(par.N):
    if np.nonzero(x[0,:,k])[0] != []:
        order[k] = np.nonzero(x[0,:,k])[0][0]

fig = plt.figure(figsize=(7,4), dpi=300)
plt.pcolormesh(x[0,:,np.argsort(order)],cmap='Greys')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xticks(np.arange(par.T)[::2000],np.linspace(0,par.T*par.dt,par.T)[::2000].round(0))
plt.xlabel('time [ms]')
plt.ylabel('inputs')
plt.savefig(par.dir+'pattern_sorted.png',format='png', dpi=300)

#%%
#plt.imshow(w[:,np.argsort(order)].T,aspect='auto')

fig = plt.figure(figsize=(7,4), dpi=300)
plt.pcolormesh(w[:,np.argsort(order)].T,cmap='coolwarm')
plt.colorbar()
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel(r'$\vec{w}$')
plt.savefig(par.dir+'w_sorted.png',format='png', dpi=300)

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

