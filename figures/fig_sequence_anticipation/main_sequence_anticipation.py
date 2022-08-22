"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequence_anticipation.py"
single neuron trained on input sequences - Fig 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    
from predictive_neuron import models, funs

'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(neuron.v.detach().numpy())              

        if par.optimizer == 'online':            
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online(par.hardbound)            

        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'-------------------'
def train(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'create input data'
    if par.spk_volley == 'deterministic':
        timing = np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt
    if par.spk_volley == 'random':
        timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
    x_data, density = funs.get_sequence(par,timing)
        
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
    if par.optimizer == 'Adam':
        optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    elif par.optimizer == 'SGD':
        optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
    
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
        if par.optimizer != "online":
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
'-------------------'


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    single neuron trained on sequences
                    """
                    )
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['online','SGD','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--hardbound',type=str,default='False',
                        help='set hard lower bound for parameters')
    parser.add_argument('--init',type=str, 
                        choices=['classic','trunc_gauss','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--init_mean',type=float, default=0.2)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.4)
    parser.add_argument('--w_0',type=float, default=.03,
                        help='fixed initial condition')
    parser.add_argument('--eta',type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches')
    parser.add_argument('--rep', type=int, default=1)   
    'input sequence'
    parser.add_argument('--spk_volley',type=str, 
                        choices=['deterministic','random'],default='random',
                        help='type of spike volley')
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N', type=int, default=100) 
    ## REMEMBER THAT YOU WANT TO ADD THE DISTRACTORS AGAIN
    parser.add_argument('--fr_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=.01) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2) 
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    'additional parameters'
    par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/fig2/'
    par.device = "cuda" if torch.cuda.is_available() else "cpu"
    par.tau_x = 2.
    par.T = int((par.Dt*par.N)/(2*par.dt))
    
    loss, w, v, spk = train(par)
    
    '--------------------'
    
    'output spike times'
    fig = plt.figure(figsize=(5,6), dpi=300)
    for k,j in zip(spk,range(par.epochs)):
        plt.scatter([j]*len(k),k,edgecolor='royalblue',facecolor='none',s=7)
    plt.xlabel(r'epochs')
    plt.xlim(0,2000)
    plt.ylabel('spk times [ms]')
    plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig('spk_sequence.png',format='png', dpi=300)
    plt.close('all')
    
    'weights dynamics'
    fig = plt.figure(figsize=(4,5), dpi=300)    
    plt.title(r'$\vec{w}$')
    plt.pcolormesh(w.T,cmap='coolwarm',norm=MidpointNormalize(midpoint=1))
    plt.colorbar()
    plt.ylabel('inputs')
    plt.xlabel(r'epochs')
    plt.xlim(0,2000)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig('w_sequence.png',format='png', dpi=300)
    plt.close('all')
