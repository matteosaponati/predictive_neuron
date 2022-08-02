"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_selforganization_lateral.py"
neural network with self-organization lateral connections

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn as nn
import numpy as np

from predictive_neuron import models_nn, funs

'----------------'
def forward(par,network,x_data):
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    for t in range(par.T):     
        'append voltage state'
        v.append(network.v.clone().detach().numpy())
        'update weights online'
        if par.online == True: 
            with torch.no_grad():
                network.backward_online(x_data[:,t])
                network.update_online()  
        'forward pass'
        network(x_data[:,t]) 
        'append output spikes'
        for n in range(par.nn):
            for b in range(par.batch):
                if network.z[b,n] != 0: z[n][b].append(t*par.dt)          
        
    return network, np.stack(v,axis=1), z
'----------------'

def train(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'create input data'
    x_data = funs.get_sequence_nn_selforg(par,random=par.random)
    
    'set model'
    network = models_nn.NetworkClass_SelfOrg(par)
    
    'initialization'
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.n_in+par.lateral,par.nn)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=.1/np.sqrt(par.par.n_in+par.lateral),
                                    a=par.init_a,b=par.init_b) 
        network.w[par.n_in:,] = par.w_0rec    
    if par.init == 'fixed':
        w = par.w_0*torch.ones(par.n_in+par.lateral,par.nn)
        w[par.n_in:,] = par.w_0rec
        network.w = nn.Parameter(w).to(par.device)
    
    'allocate outputs'
    w = np.zeros((par.epochs,par.n_in+par.lateral,par.nn))
    z_out = [[] for n in range(par.nn)]
    v_out = []
    
    for e in range(par.epochs):
            
        network.state()
        network, v, z = forward(par,network,x_data)
            
        v_out.append(v)
        
        w[e,:,:] = network.w.detach().numpy()
        for n in range(par.nn):
            z_out[n].append(z[n])
        
        if e%50 == 0: print(e)  

    return w, z_out, v_out
    
'-------------------'

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    single neuron trained on spike patterns
                    """
                    )
    'initialization'
    parser.add_argument('--init',type=str, 
                        choices=['classic','trunc_gauss','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--init_mean',type=float, default=0.01)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.0)
    parser.add_argument('--w_0',type=float, default=.08,
                        help='fixed initial condition')
    parser.add_argument('--w_0rec',type=float, default=.0003,
                        help='fixed initial condition')
    'optimizer'
    parser.add_argument('--online', type=bool, default=True,
                        help='set online learning algorithm')
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches') 
    parser.add_argument('--eta',type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=1992)
      
    'setup inputs'
    parser.add_argument('--n_in', type=int, default=2)
    parser.add_argument('--delay', type=int, default=4)
    parser.add_argument('--Dt', type=int, default=2)
    parser.add_argument('--random', type=bool, default=True) 

    'neuron model'
    parser.add_argument('--is_rec', type=bool, default=True,
                        help='set recurrent connections')
    parser.add_argument('--nn', type=int, default=6)
    parser.add_argument('--lateral', type=int, default=2)
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 20.) 
    parser.add_argument('--v_th', type=float, default= 3.)
    parser.add_argument('--dtype', type=str, default=torch.float)
    
    parser.add_argument('--savedir', type=str, default='') 
    
    par = parser.parse_args()
    'additional parameters'
    # par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/patterns/'
    par.device = "cpu"
    par.tau_x = 2.
    par.random = False
    par.T = int((par.nn*par.delay+par.Dt+70)/par.dt)
    
    w, spk, v = train(par)
    
    np.save(par.savedir+'w_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}'.format(
                            par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th),w)
    # np.save(par.savedir+'v_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}'.format(
    #                         par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th),v)
    # np.save(par.savedir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}'.format(
    #                         par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th),spk)
    
