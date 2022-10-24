"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequence.py"
single neuron trained on input sequences - Figure 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import numpy as np
    
from predictive_neuron import models, funs, funs_train

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description='single neuron trained on sequences'
                    )
    
    parser.add_argument('--name',type=str, 
                        choices=['sequence','multisequence'],default='sequence',
                        help='type of sequence inputs')
    
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['online','SGD','Adam'],default='Adam',
                        help='choice of optimizer')
    
    parser.add_argument('--bound',type=str,
                        choices=['None','hard','soft'],default='None',
                        help='set hard lower bound for parameters')
    
    parser.add_argument('--init',type=str, 
                        choices=['classic','random','fixed'],default='fixed',
                        help='type of weights initialization')
    
    parser.add_argument('--init_mean',type=float, default=0.1)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.2)

    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--rep', type=int, default=1)
    
    'set input sequence'
    parser.add_argument('--sequence',type=str, 
                        choices=['deterministic','random'],default='random')
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N_seq', type=int, default=10)
    parser.add_argument('--N_dist', type=int, default=10)
    
    'set noise sources'
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2)
    parser.add_argument('--onset', type=bool, default=False)
    
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--eta',type=float, default=1e-3)
    parser.add_argument('--tau_x', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    
    '-----------------'
    
    'set total length of simulation and total input size'
    par.T = int(2*(par.Dt*par.N_seq + par.jitter)/(par.dt))
    par.N = par.N_seq+par.N_dist
    
    'set timing'
    if par.sequence == 'deterministic':
        timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)
    if par.sequence == 'random':
        timing = (np.cumsum(np.random.randint(0,par.Dt,par.N_seq))/par.dt).astype(int)
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
        
    'set model'
    neuron = models.NeuronClass(par)
    neuron = funs_train.initialize_weights_PyTorch(par,neuron)
    
    if par.noise == False:
        x_data, onset = funs.get_sequence(par,timing)
        w,v,spk,loss = funs_train.train_PyTorch(par,neuron,x=x_data)
    
    else: 
        w,v,spk,loss = funs_train.train_PyTorch(par,neuron,timing=timing)
        
    # par.savedir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences/'
    # par.loaddir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences/'
    
    # np.save('v_tau_{}_vth_{}_rep_{}'.format(par.tau_m,par.v_th,par.rep),v)
    # np.save('w_tau_{}_vth_{}_rep_{}'.format(par.tau_m,par.v_th,par.rep),w)
    # np.save('spk_tau_{}_vth_{}_rep_{}'.format(par.tau_m,par.v_th,par.rep),spk)
    # np.save('loss_tau_{}_vth_{}_rep_{}'.format(par.tau_m,par.v_th,par.rep),loss)
    
    '--------------------'
