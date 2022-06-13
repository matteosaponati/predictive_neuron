"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"capacity_main.py"
predictive processes at the single neuron level - analysis of 
model capacity as a function of the number of synapses

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn as nn
import numpy as np

from predictive_neuron import models, funs

import torch.nn.functional as F
def sequence_capacity(par,timing):
    
    'create sequence'
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)   
    for k in range(par.batch):
        x_data[k,timing,k*par.N_sub + np.arange(par.N_sub)] = 1

    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)

'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], [[] for k in range(par.batch)]
    
    for t in range(par.T):            

        v.append(neuron.v)              

        neuron(x_data[:,t])     
        for k in range(par.batch):
            if neuron.z[k] != 0: z[k].append(t*par.dt)    
        
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
    v_out, spk_out = [], [[] for k in range(par.batch)]
    
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
        for k in range(par.batch):
            spk_out[k].append(z[k])
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return loss_out, w, v_out, spk_out


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    single neuron trained on sequences
                    """
                    )
    
    parser.add_argument('--type',type=str, 
                        choices=['overlap','no-overlap'],default='no-overlap',
                        help='type of analysis')
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['online','SGD','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--spk_volley',type=str, 
                        choices=['fixed','random'],default='fixed',
                        help='type of input sequence')
    parser.add_argument('--init',type=str, 
                        choices=['classic','trunc_gauss','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--w_0',type=float, default=.03,
                        help='fixed initial condition')
    parser.add_argument('--eta',type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=1992)
    'input sequence'
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N', type=int, default=100) 
    parser.add_argument('--N_sub', type=int, default=2) 
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    'additional parameters'
    par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/capacity/'
#    par.device = "cuda" if torch.cuda.is_available() else "cpu"
    par.device = "cpu"
    par.tau_x = 2.
    
    par.T = int((2*par.N_sub+10) // par.dt)
    par.batch = int(par.N/par.N_sub)
    
    timing = np.linspace(par.Dt,par.Dt*par.N_sub,par.N_sub)/par.dt
    x_data = sequence_capacity(par,timing)
    
    w, spk = train(par,x_data)
    
    np.save(par.savedir+'w_N_{}_th_{}_tau_{}'.format(par.N_sub,par.v_th,par.tau_m),w)
    np.save(par.savedir+'spk_N_{}_th_{}_tau_{}'.format(par.N_sub,par.v_th,par.tau_m),spk)
