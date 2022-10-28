"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_multisequence_PyTorch.py"
neural network with recurrent inhibition

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
    
from predictive_neuron import models, funs, funs_train_inhibition


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description='neural network trained on multiple sequences'
                    )
    
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['online','SGD','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--bound',type=str,
                        choices=['None','hard','soft'],default='None',
                        help='set hard lower bound for parameters')
    parser.add_argument('--init',type=str, 
                        choices=['random','fixed'],default='random',
                        help='type of weights initialization')
    
    parser.add_argument('--init_mean',type=float, default=0.1)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.2)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--rep', type=int, default=1)
    
    'set input sequence'
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--Dt', type=int, default=2) 
    
    'set noise sources'
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2)
    
    'network model'
    parser.add_argument('--is_rec', type=bool, default=True)
    parser.add_argument('--nn', type=int, default=10)
    parser.add_argument('--w0_rec', type=float, default=-.05) 
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--eta',type=float, default=1e-3)
    parser.add_argument('--tau_x', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    parser.add_argument('--upload_data', type=bool, default=False)  
    parser.add_argument('--load_dir', type=str, default='') 
    parser.add_argument('--save_dir', type=str, default='')
    
    par = parser.parse_args()
    
    '-----------------'
    
    'set total length of simulation'
    par.T = int(2*(par.Dt*par.n_in + par.jitter)/(par.dt))
    
    'set timing'
    spk_times = []
    for b in range(par.batch):
        spk_times.append(np.random.shuffle( \
                            (np.linspace(
                                    par.Dt,par.Dt*par.n_in,par.n_in)/par.dt).astype(int)))
    timing = [[] for n in range(par.nn)]
    for n in range(par.nn):
        for b in range(par.batch): timing[n].append(spk_times[b])
        
    'set model'
    network = models.NetworkClass(par)
    network = funs_train_inhibition.initialize_weights_nn_PyTorch(par,network)
    
    print(par.load_dir)
    if par.noise == True:
        
        if par.upload_data == True: 
            w, v, spk = funs_train_inhibition.train_nn_PyTorch(par,network)
        else: 
            w, v, spk = funs_train_inhibition.train_nn_PyTorch(par,network,timing=timing)

    else:
        x = funs.funs.get_multisequence_nn(par,timing)
        w, v, spk = funs_train_inhibition.train_nn_PyTorch(par,network,x=x)
        
    np.save(par.save_dir+'w_taum_{}_vth_{}_eta_{}_init_mean_{}_wrec_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w_0rec),w)
    np.save(par.save_dir+'v_taum_{}_vth_{}_eta_{}_init_mean_{}_wrec_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w_0rec),v)
    np.save(par.save_dir+'spk_taum_{}_vth_{}_eta_{}_init_mean_{}_wrec_{}'.format(
                            par.tau_m,par.v_th,par.eta,par.init_mean,par.w_0rec),spk)
