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

'training algorithm'
parser.add_argument('--optimizer',type=str, 
                    choices=['SGD','Adam'],default='Adam',
                    help='choice of optimizer')
parser.add_argument('--l_rate',type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs')
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--batch', type=int, default=1,
                    help='number of batches')   
'architecture'
parser.add_argument('--T', type=int, default=400) 
parser.add_argument('--timing', type=list, default=[2.,6.],
                    help='sequence timing') 
'neuronal model'
parser.add_argument('--dt', type=float, default= .05) 
parser.add_argument('--tau_m', type=float, default= 10.) 
parser.add_argument('--v_th', type=float, default= 2.5)
parser.add_argument('--dtype', type=str, default=torch.float) 

par = parser.parse_args()

'additional parameters'
par.device = "cuda" if torch.cuda.is_available() else "cpu"
'input'
par.N = len(par.timing)
par.tau_x = 2.

'train'
E, w1, w2, v, spk =  train.train(par)

'-----------'

print('saving')
np.save('data/loss_rep_{}'.format(par.rep),E)
np.save('data/w1_rep_{}'.format(par.rep),w1)     
np.save('data/w2_rep_{}'.format(par.rep),w2)  
np.save('data/v_rep_{}'.format(par.rep),v)
np.save('data/spk_rep_{}'.format(par.rep),spk)







