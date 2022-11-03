"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"sequence_create_dataset.py"
single neuron trained on input sequences - Fig 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import os 

from predictive_neuron import funs

'-------------------'

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    'general'
    parser.add_argument('--type', type=str, 
                        choices=['NumPy','PyTorch'], default='NumPy')
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--dtype', type=str, default=torch.float) 
    parser.add_argument('--rep', type=int, default=2)
    
    'set input sequence'
    parser.add_argument('--sequence',type=str, 
                        choices=['deterministic','random'],default='deterministic')
    parser.add_argument('--Dt', type=int, default=2) 
    parser.add_argument('--N_seq', type=int, default=100)
    parser.add_argument('--N_dist', type=int, default=100)
    
    parser.add_argument('--tau_x', type=float, default= 2.)
    parser.add_argument('--dt', type=float, default= .05) 
    
    'noise sources'
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--freq_noise', type=int, default=0)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=int, default=0) 
    parser.add_argument('--jitter', type=float, default=2)
    parser.add_argument('--onset', type=int, default=0)
    
    par = parser.parse_args()

    'set total length of simulation and total input size'
    par.T = int(2*(par.Dt*par.N_seq + par.jitter)/(par.dt))
    par.N = par.N_seq+par.N_dist
    
    'set onset'
    onset = None
    if par.onset == 1:
        onset = np.random.randint(0,par.T/2)
    
    'set timing'
    if par.sequence == 'deterministic':
        timing = (np.linspace(par.Dt,par.Dt*par.N_seq,
                              par.N_seq)/par.dt).astype(int)
    if par.sequence == 'random':
        timing = (np.cumsum(np.random.randint(0,par.Dt,
                            par.N_seq))/par.dt).astype(int)
        
    'set directories'
    savedir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences_dataset/'+ \
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                	par.Dt,par.N_seq,par.N_dist,par.noise,par.jitter_noise,
                    par.jitter,par.freq_noise,par.freq,par.onset)
    print(savedir)
    if not os.path.exists(savedir): 
        os.makedirs(savedir)
    
    'create datapoint'
    if par.type == 'NumPy':
        x = funs.get_sequence_NumPy(par,timing,onset=onset)
        np.save(savedir+'x_NumPy_{}'.format(par.rep),x)
    if par.type == 'PyTorch':
        x = funs.get_sequence(par,timing,onset=onset)
        np.save(savedir+'x_PyTorch_{}'.format(par.rep),x)