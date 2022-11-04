"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequence_NumPy.py"
single neuron trained on input sequences - Figure 2 (NumPy version)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import os
    
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
    parser.add_argument('--bound',type=str,default='soft',
                        help='set hard lower bound for parameters')
    
    parser.add_argument('--init',type=str, 
                        choices=['classic','random','fixed'],default='fixed',
                        help='type of weights initialization')
    
    parser.add_argument('--init_mean',type=float, default=0.1)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.05)

    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--rep', type=int, default=1)
    
    'set input sequence'
    parser.add_argument('--sequence',type=str, 
                        choices=['deterministic','random'],default='deterministic')
    parser.add_argument('--Dt', type=int, default=2) 
    parser.add_argument('--N_seq', type=int, default=100)
    parser.add_argument('--N_dist', type=int, default=100)
    
    'noise sources'
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--freq_noise', type=int, default=0)
    parser.add_argument('--freq', type=float, default=10.) 
    parser.add_argument('--jitter_noise', type=int, default=0) 
    parser.add_argument('--jitter', type=float, default=2.)
    parser.add_argument('--onset', type=int, default=0)
    
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 1.4)
    parser.add_argument('--eta',type=float, default=5e-4)
    parser.add_argument('--tau_x', type=float, default= 2.)
    
    parser.add_argument('--upload_data', type=int, default=1)  
    parser.add_argument('--load_dir', type=str, default='') 
    parser.add_argument('--save_dir', type=str, default='')
    
    par = parser.parse_args()
    
    '-----------------'
    
    par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences/'+\
		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                    par.	Dt,par.N_seq,par.N_dist,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq,par.onset)
    if not os.path.exists(par.save_dir): os.makedirs(par.save_dir)
    
    par.load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/sequences_dataset/'+ \
    		'Dt_{}_N_seq_{}_N_dist_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}_onset_{}/'.format(
                	par.Dt,par.N_seq,par.N_dist,par.noise,par.jitter_noise,
                    par.jitter,par.freq_noise,par.freq,par.onset)
    
    'set total length of simulation and total input size'
    par.T = int(2*(par.Dt*par.N_seq + par.jitter)/(par.dt))
    par.N = par.N_seq+par.N_dist
    
    'set onset'
    if par.onset == 1:
        par.onset_list = np.random.randint(0,par.T/2,par.epochs)
        np.save(par.save_dir+'onset_list_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}'.format(
                        par.tau_m,par.v_th,par.eta,par.init_mean,par.rep),par.onset_list)
    
    'set timing'
    if par.sequence == 'deterministic':
        timing = (np.linspace(par.Dt,par.Dt*par.N_seq,par.N_seq)/par.dt).astype(int)
    if par.sequence == 'random':
        timing = (np.cumsum(np.random.randint(0,par.Dt,par.N_seq))/par.dt).astype(int)
        
    'set model'
    neuron = models.NeuronClass_NumPy(par)
    neuron.w = funs_train.initialize_weights_NumPy(par,neuron)
    
    'train'
    if par.noise == 0:
        x = funs.get_sequence_NumPy(par,timing)
        w,v,spk,loss = funs_train.train_NumPy(par,neuron,x=x)
    
    else: 
        w,v,spk,loss = funs_train.train_NumPy(par,neuron,timing=timing)
        
    'save'
    np.save(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.rep),w)
    np.save(par.save_dir+'v_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.rep),v)
    np.save(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_rep_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.rep),spk)
    
    '--------------------'