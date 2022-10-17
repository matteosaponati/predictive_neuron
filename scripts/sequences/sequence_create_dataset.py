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

import torch
import numpy as np
    
from predictive_neuron import funs

'-------------------'

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    single neuron trained on sequences
                    """
                    )
    
    'training algorithm'
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--rep', type=int, default=2)
    
    'input sequence'
    parser.add_argument('--sequence',type=str, 
                        choices=['deterministic','random'],default='random',
                        help='type of spike volley')
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N_seq', type=int, default=10)
    parser.add_argument('--N_dist', type=int, default=15)
    parser.add_argument('--offset', type=bool, default=False)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2) 
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    'additional parameters'
    par.savedir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequence_dataset/'
    par.device = "cpu"
    par.tau_x = 2.
    
    'set total length of simulation'
    par.T = int((par.Dt*par.N_seq)/(par.dt))
    'set total input'
    par.N = par.N_seq+par.N_dist    
    
    'create input data'
    if par.sequence == 'deterministic':
        timing = np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt
    if par.sequence == 'random':
        timing = np.cumsum(np.random.randint(0,par.Dt,par.N_seq))/par.dt
    'set total number of inputs: sequence + distractors'
    x_data = funs.get_sequence(par,timing)
    
    np.save(par.savedir+'sequence_Dt_{}_Nseq_{}_Ndist_{}_rep_{}'.format(par.Dt,par.N_seq,par.N_dist,par.rep),x_data)