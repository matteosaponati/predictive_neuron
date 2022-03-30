"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"fig1_main.py"
predictive processes at the single neuron level - single neuron model
trained on input sequences

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import numpy as np
import argparse

import train

parser = argparse.ArgumentParser(
                    description="""
                    reproduction of results in Figure 1
                    """
                    )

'options'
parser.add_argument('--online',type=str,default='False',
                    help='train mode with online approx algorithm')
parser.add_argument('--hardbound',type=str,default='False',
                    help='set hard lower bound for parameters')
parser.add_argument('--rep',type=int,default=1)
parser.add_argument('--simulation',type=str, 
                    choices=['classic','par_sweep','ic'],default='classic',
                    help='type of simulation')
parser.add_argument('--init',type=str, 
                    choices=['classic','trunc_gauss','fixed'],default='classic',
                    help='type of weights initialization')

'training algorithm'
parser.add_argument('--optimizer',type=str, 
                    choices=['None','SGD','Adam'],default='Adam',
                    help='choice of optimizer')
parser.add_argument('--l_rate',type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs')
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--batch', type=int, default=1,
                    help='number of batches')   
'architecture'
parser.add_argument('--T', type=int, default=300) 
parser.add_argument('--timing', type=float, default=4.,
                    help='delay between inputs') 
'neuronal model'
parser.add_argument('--dt', type=float, default= .05) 
parser.add_argument('--tau_m', type=float, default= 10.) 
parser.add_argument('--v_th', type=float, default= 2.5)
parser.add_argument('--dtype', type=str, default=torch.float)
parser.add_argument('--w_0', type=float, default= .03) 

par = parser.parse_args()

'additional parameters'
par.device = "cpu"
'input'
par.N = 2
par.tau_x = 2.

'train'
E, w1, w2, v, spk =  train.train(par)

'-----------'

print('saving')
savedir = '/mnt/hpx/slurm/saponatim/predictive_neuron/figures/fig1/data/'

if par.simulation == 'ic':
    np.save(savedir+'loss_rep_{}'.format(par.rep),E)
    np.save(savedir+'w1_rep_{}'.format(par.rep),w1)     
    np.save(savedir+'w2_rep_{}'.format(par.rep),w2)  
    np.save(savedir+'v_rep_{}'.format(par.rep),v)
    np.save(savedir+'spk_rep_{}'.format(par.rep),spk)
    
if par.simulation == 'classic':
    np.save(savedir+'loss',E)
    np.save(savedir+'w1',w1)     
    np.save(savedir+'w2',w2)  
    np.save(savedir+'v',v)
    np.save(savedir+'spk',spk)
    
if par.simulation == 'par_sweep':
    np.save(savedir+'loss_dt_{}_tau_{}'.format(par.timing,par.tau_m),E)
    np.save(savedir+'w1_dt_{}_tau_{}'.format(par.timing,par.tau_m),w1)     
    np.save(savedir+'w2_dt_{}_tau_{}'.format(par.timing,par.tau_m),w2)  
    np.save(savedir+'v_dt_{}_tau_{}'.format(par.timing,par.tau_m),v)
    np.save(savedir+'spk_dt_{}_tau_{}'.format(par.timing,par.tau_m),spk)
    