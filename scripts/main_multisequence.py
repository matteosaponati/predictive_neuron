"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_multisequence.py"
single neuron trained on input sequences (spike volleys) with multiple
subcomponents

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
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(neuron.v)              

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
    timing = []
    for k  in range(par.sequences):
        
        if par.spk_volley == 'deterministic':
            sequence = np.linspace(par.Dt,par.Dt*par.N_sequences[k],
                                     par.N_sequences[k])
        if par.spk_volley == 'random':
            sequence = np.cumsum(np.random.randint(0,par.Dt,
                                               len(par.N_sequences[k])))
        timing.append(k*(par.Dt*len(par.N_sequences[k]))/par.dt + 
                      sequence/par.dt)
    
    x_data, density = funs.get_multi_sequence(par,timing)
    
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
    parser.add_argument('--init_mean',type=float, default=0.05)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.1)
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
    parser.add_argument('--spk_volley',type=str, 
                        choices=['deterministic','random'],default='deterministic',
                        help='type of spike volley')
    parser.add_argument('--sequences', type=int, default=2,
                        help = 'number of sub-sequences') 
    parser.add_argument('--N_sub', type=int, default=2,
                        help='number of input in subsequence')
    parser.add_argument('--Dt', type=int, default=4,
                        help='latency between inputs in sequence')
    parser.add_argument('--DT', type=int, default=10,
                        help='delay between subsequences') 
    
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    
    'additional parameters'
    par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'
#    par.device = "cuda" if torch.cuda.is_available() else "cpu"
    par.device = "cpu"
    par.tau_x = 2.
    
    par.N_sequences = [par.N_sub*k + np.arange(par.N_sub,dtype=int) for k in range(par.sequences)]
    par.N = np.sum(par.N_sub*par.sequences,dtype=int)
    par.T = int(((par.Dt*par.N/par.dt)))
    par.DT = int(par.DT/par.dt)
    
    loss, w, v, spk = train(par)
    
    np.save(par.savedir+'loss_seq_{}_N_{}_vth_{}_tau_{}'.format(par.sequences,par.N_sub,par.v_th,par.tau_m),loss)
    np.save(par.savedir+'w_seq_{}_N_{}_vth_{}_tau_{}'.format(par.sequences,par.N_sub,par.v_th,par.tau_m),w)
    np.save(par.savedir+'v_seq_{}_N_{}_vth_{}_tau_{}'.format(par.sequences,par.N_sub,par.v_th,par.tau_m),v)
    np.save(par.savedir+'spk_seq_{}_N_{}_vth_{}_tau_{}'.format(par.sequences,par.N_sub,par.v_th,par.tau_m),spk)
    