"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"tools.py"
auxiliary functions to reproduce results of Figure 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import torch.nn as nn

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


from predictive_neuron import models, funs

import matplotlib.colors as colors

'------------------------------'
'------------------------------'
'optimization '

'----------------'
def get_input(par):
    
    if par.input == 'sequence':
         x_data,density,fr = funs.get_sequence(par,par.timing)
         return x_data
    
    if par.input == 'pattern':
         x_data,density,fr = funs.get_pattern(par)
         return x_data
        
'----------------'

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
    
    'input type'
    if par.input == 'sequence':
        if par.spk_volley == 'deterministic':
            par.timing = np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt
        if par.spk_volley == 'random':
            par.timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
    if par.input == 'pattern':
        par.mask = torch.rand(par.batch,par.T_pattern,par.N).to(par.device)
        
    x_data = get_input(par)
    
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
        
        x_data = get_input(par)
        
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


'------------------------------'
'------------------------------'
'plots'

def sequence_example(par,savedir):
    
    '1. sequence example'

    fig = plt.figure(figsize=(4,4), dpi=300)
    timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
    timing += np.random.randint(0,par.T/2) + np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,par.N)
    offset = 1
    for k in range(par.N):
        bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
        plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
        plt.eventplot([timing[k]],lineoffsets = offset,linelengths = 3,linewidths = .5,colors = 'purple')
        offset += 1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'spk_volley.png',format='png', dpi=300)
    plt.savefig(savedir+'spk_volley.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
def capacity_example(par,savedir,delay=40):
    
    '1. capacity example'
    
    timing = []
    for k in range(par.batch):
        timing.append((k*delay + np.cumsum(np.random.randint(0,par.Dt,par.N_sub)))/par.dt)
    timing = np.array(timing).flatten()
    par.T = int((par.batch*(par.N_sub*par.Dt + delay)/2)/par.dt)

    fig = plt.figure(figsize=(4,4), dpi=300)
    offset = 1
    for k in range(par.N):
        bg = (np.where(np.random.rand(par.T)<(par.freq*par.dt))[0]).astype(int)
        plt.eventplot(bg,lineoffsets = offset,linelengths = 3,linewidths = .5,alpha=1,colors = 'grey')
        plt.eventplot([timing[k]],lineoffsets = offset,linelengths = 3,linewidths = .5,colors = 'mediumvioletred')
        offset += 1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel('inputs')
    plt.savefig(savedir+'spk_capacity.png',format='png', dpi=300)
    plt.savefig(savedir+'spk_capacity.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    
def plot_w_spk(par,w,spk):    
    
    '4. weights dynamics'
    hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
    fig = plt.figure(figsize=(8,4), dpi=300)    
    divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
    plt.imshow(np.flip(w.T,axis=0),cmap=funs.get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.ylabel('inputs')
    plt.xlabel(r'epochs')
    plt.savefig('w_{}.png'.format(par.input),format='png', dpi=300)
    plt.close('all')
    
    '4. output spikes'
    spk_eff = []
    for k in range(len(spk)):
         spk_eff.append(spk[k][::2])
    fig = plt.figure(figsize=(8,4), dpi=300)
    for k,j in zip(spk,range(par.epochs)):
        plt.scatter([j]*len(k),k,edgecolor='navy',facecolor='none',s=1)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel(r'epochs')
    plt.xlim(0,2000)
    plt.ylim(0,200)
    plt.ylabel('spk times [ms]')
    plt.grid(True,which='both',color='darkgrey',linewidth=.7)
    plt.savefig('spk_{}.png'.format(par.input),format='png', dpi=300)
    plt.close('all')
        
def seq_density(par,timing,savedir):
    
    '2. density'
    timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
    'add background firing noise'
    prob = par.freq*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x[mask<prob] = 1        
    x[:,timing,range(par.N)] = 1
    'get density'
    bins = np.arange(par.T).tolist()
    step = int(par.tau_m/par.dt)
    bins = [bins[i:i+step] for i in range(0,len(bins),int(1/par.dt))]
    density = [torch.sum(x[0,bins[k],:]).item() for k in range(len(bins))]
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.plot(density,linewidth=2,color='mediumvioletred')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel(r'density [spk/$\tau_m$]')
    plt.savefig(savedir+'seq_density.png',format='png', dpi=300)
    plt.savefig(savedir+'seq_density.pdf',format='pdf', dpi=300)
    plt.close('all')
    
def seq_fr_average(par,savedir,repetitions=500):
    
    '3. average firing rate'
    fr_avg = []
    for k in range(repetitions):
        timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
        _, fr = funs.get_sequence_density(par,timing,mu=True,jitter=True,offset=True)
        fr_avg.append(fr)
    fr_m = np.mean(np.array(fr_avg),axis=0)
    fr_std = np.std(np.array(fr_avg),axis=0)
    
    fig = plt.figure(figsize=(4,4), dpi=300)
    plt.plot(fr_m,linewidth=2,color='k')
    plt.fill_between(range(399),fr_m + .5*fr_std, fr_m - .5*fr_std,alpha=.5,color='grey')
    plt.ylim(0,30)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.ylabel(r'firing rate [Hz]')
    plt.savefig(savedir+'seq_fr.png',format='png', dpi=300)
    plt.savefig(savedir+'seq_fr.pdf',format='pdf', dpi=300)
    plt.close('all')

