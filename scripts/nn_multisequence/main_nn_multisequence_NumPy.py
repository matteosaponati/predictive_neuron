"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_multisequence_NumPy.py"
neural network with recurrent inhibition

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import os
    
from predictive_neuron import funs, funs_train_inhibition

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description='neural network trained on multiple sequences'
                    )
    
    'training algorithm'
    parser.add_argument('--bound',type=str,
                        choices=['None','hard','soft'],default='None',
                        help='set hard lower bound for parameters')
    parser.add_argument('--init',type=str, 
                        choices=['uniform','fixed'],default='uniform',
                        help='type of weights initialization')
    
    parser.add_argument('--init_mean',type=float, default=0.03)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.2)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--rep', type=int, default=1)
    
    'set input sequence'
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--nn', type=int, default=2)
    parser.add_argument('--Dt', type=int, default=2) 
    
    'set noise sources'
    parser.add_argument('--noise', type=int, default=0)
    parser.add_argument('--freq_noise', type=int, default=0)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=int, default=0) 
    parser.add_argument('--jitter', type=float, default=2)
    
    'network model'
    parser.add_argument('--is_rec', type=bool, default=0)
    parser.add_argument('--w0_rec', type=float, default=-.05) 
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--eta',type=float, default=4e-5)
    parser.add_argument('--tau_x', type=float, default= 2.)
    
    parser.add_argument('--upload_data', type=bool, default=0)  
    parser.add_argument('--load_dir', type=str, default='') 
    parser.add_argument('--save_dir', type=str, default='')
    
    par = parser.parse_args()
    
    '-----------------'
    
    par.save_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy/'+\
		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                    par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq)
    if not os.path.exists(par.save_dir): os.makedirs(par.save_dir)
    
    par.load_dir = '/mnt/hpc/departmentN4/predictive_neuron_data/nn_multisequence_NumPy_dataset/'+\
		'Dt_{}_nn_{}_N_{}_batch_{}_noise_{}_jitter_noise_{}_jitter_{}_freq_noise_{}_freq_{}/'.format(
                    par.Dt,par.nn,par.N,par.batch,par.noise,par.jitter_noise,par.jitter,
                    par.freq_noise,par.freq)
        
    'set total length of simulation'
    par.T = int(2*(par.Dt*par.N + par.jitter)/(par.dt))
    
    'set model'
    network = funs_train_inhibition.initialize_nn_NumPy(par)
    
    'train'
    if par.noise == 0:
        'set timing'
        spk_times = []
        for b in range(par.batch):
            times = (np.linspace(par.Dt,par.Dt*par.N,par.N)/par.dt).astype(int)
            np.random.shuffle(times)
            spk_times.append(times)
        timing = [[] for n in range(par.nn)]
        for n in range(par.nn):
            for b in range(par.batch): timing[n].append(spk_times[b])
        np.save(par.save_dir+'timing_taum_{}_vth_{}_eta_{}_w0_rec_{}_init_mean_{}_rep_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.w0_rec,par.init_mean,par.rep),timing)
        'get inputs'
        x = funs.get_multisequence_nn_NumPy(par,timing)
        'train network'
        w,spk,loss = funs_train_inhibition.train_nn_NumPy(par,network,x=x)
    
    else: 
        w,spk,loss = funs_train_inhibition.train_nn_NumPy(par,network,timing=timing)
        
    np.save(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,par.rep),w)
#    np.save(par.save_dir+'v_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}'.format(
#                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,par.rep),v)
    np.save(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_w0_rec_{}_rep_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w0_rec,par.rep),spk)