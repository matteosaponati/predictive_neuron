"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_selforganization_NumPy.py"
neural network with self-organization lateral connections -- NumPy version

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
from predictive_neuron import models, funs_train

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    """
                    )
    
    'optimization'
    parser.add_argument('--batch', type=int, default=1) 
    parser.add_argument('--eta',type=float, default=1e-5)
    parser.add_argument('--bound',type=str, default='none')
    parser.add_argument('--epochs', type=int, default=10)
    
    'initialization'
    parser.add_argument('--init',type=str, 
                        choices=['random','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--init_mean',type=float, default=0.02)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.08)
    parser.add_argument('--w_0rec',type=float, default=.0003,
                        help='fixed initial condition')
      
    'inputs'
    parser.add_argument('--n_in', type=int, default=26)
    parser.add_argument('--delay', type=int, default=4)
    parser.add_argument('--Dt', type=int, default=2)
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=.01) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2)  

    'network model'
    parser.add_argument('--is_rec', type=bool, default=True,
                        help='set recurrent connections')
    parser.add_argument('--nn', type=int, default=8)
    parser.add_argument('--lateral', type=int, default=2)
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 15.) 
    parser.add_argument('--v_th', type=float, default= 3.5)
    parser.add_argument('--tau_x', type=float, default= 2.)
   
    parser.add_argument('--upload_data', type=bool, default=True)  
    parser.add_argument('--load_dir', type=str, default='') 
    parser.add_argument('--save_dir', type=str, default='')
   
    par = parser.parse_args()
    
    '-------------'
    
    'set simulation length'
    par.T = int((par.nn*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)
#    par.T = int((par.nn*par.delay + par.n_in*par.Dt + 80)/par.dt)
    
    'set model'
    network = models.NetworkClass_SelfOrg_NumPy(par)
    network = funs_train.initialization_weights_nn_NumPy(par,network)
    
    w, v, spk = funs_train.train_nn_NumPy(par,network)
    
    np.save(par.save_dir+'w_taum_{}_vth_{}_init_mean_{}_wrec_{}'.format(
                            par.tau_m,par.v_th,par.init_mean,par.w_0rec),w)
    np.save(par.save_dir+'w_taum_{}_vth_{}_init_mean_{}_wrec_{}'.format(
                            par.tau_m,par.v_th,par.init_mean,par.w_0rec),v)
    np.save(par.save_dir+'w_taum_{}_vth_{}_init_mean_{}_wrec_{}'.format(
                            par.tau_m,par.v_th,par.init_mean,par.w_0rec),spk)