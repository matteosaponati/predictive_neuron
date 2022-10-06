"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig2_optimizers.py":
    
    - numerical comparison on simple optimization process 
    with spike sequence and spike pattern input.
    optimizers: online, online with hard-bound, SGD (offline), Adam (offline)

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

from predictive_neuron import models, funs

savedir = '/gs/home/saponatim/'


'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(neuron.v)              

        if par.optimizer == 'online':            
            with torch.no_grad():
                neuron.backward_online(x_data[:,t])
                neuron.update_online()            

        neuron(x_data[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z
'----------------'

'-------------------'
def train(par):
    
    'create input data'
    if par.spk_volley == 'deterministic':
        timing = np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt
    if par.spk_volley == 'random':
        timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
        
    x_data,density,fr = funs.get_sequence(par,timing)
        
    'set model'
    neuron = models.NeuronClass(par)
    loss = nn.MSELoss(reduction='sum')
    
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
    parser.add_argument('--bound',type=str,default='False',
                        help='set hard lower bound for parameters')
    parser.add_argument('--w_0',type=float, default=.03,
                        help='fixed initial condition')
    parser.add_argument('--eta',type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=4500,
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches')
    parser.add_argument('--rep', type=int, default=1)   
    'input sequence'
    parser.add_argument('--spk_volley',type=str, 
                        choices=['deterministic','random'],default='deterministic',
                        help='type of spike volley')
    parser.add_argument('--Dt', type=int, default=2) 
    parser.add_argument('--N', type=int, default=100) 
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    'additional parameters'
    par.device = "cpu"
    par.tau_x = 2.
    par.T = int((par.Dt*par.N+10)/(par.dt))
    
    'noise'
    par.freq = .01
    par.jitter = 1

    loss, w, v, spk = train(par)
    
    par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/fig2/optimizers/'
    np.save(par.savedir+'loss_opt_{}_bound_{}'.format(par.optimizer,par.hardbound),loss)
    np.save(par.savedir+'v_opt_{}_bound_{}'.format(par.optimizer,par.hardbound),v)
    np.save(par.savedir+'w_opt_{}_bound_{}'.format(par.optimizer,par.hardbound),w)
    np.save(par.savedir+'spk_opt_{}_bound_{}'.format(par.optimizer,par.hardbound),spk)
    
    'plots'
    
    savedir = '/mnt/gs/home/saponatim/'
