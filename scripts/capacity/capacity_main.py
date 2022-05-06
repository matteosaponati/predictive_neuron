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

'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], [[] for k in range(par.batch)]
    
    for t in range(par.T):            

        v.append(neuron.v)              

        if par.optimizer == 'online':            
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online(par.hardbound)            

        neuron(x_data[:,t])        
        for k in range(par.batch):
            if neuron.z[k] != 0: z[k].append(t*par.dt)  
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'-------------------'
def train(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'create input data'
    par.batch = int(par.N/2)
        
    timing = np.linspace(4,4*par.N,par.N)/par.dt
    step = int(par.N/par.batch)
    x_data = funs.get_sequence_capacity2(par,timing,step)
    
    'set model'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
    'initialization'
    if par.init == 'trunc_gauss':
        neuron.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(neuron.w, mean=0.05, std=.1/np.sqrt(par.N),
                                    a=0.,b=.1)
    if par.init == 'fixed':
        neuron.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
    
    'optimizer'
    if par.optimizer == 'Adam':
        optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    elif par.optimizer == 'SGD':
        optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)

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
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return neuron.w.detach().numpy(), z
'-------------------'

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
    parser.add_argument('--hardbound',type=str,default='False',
                        help='set hard lower bound for parameters')
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
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches')   
    'input sequence'
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N', type=int, default=2) 
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
    par.T = int((2.+(par.Dt*par.N)+50) // par.dt)
    
    w, spk = train(par)
    
    np.save(par.savedir+'w_N_{}_th_{}_tau_{}'.format(par.N,par.v_th,par.tau_m),w)
    np.save(par.savedir+'spk_N_{}_th_{}_tau_{}'.format(par.N,par.v_th,par.tau_m),spk)
