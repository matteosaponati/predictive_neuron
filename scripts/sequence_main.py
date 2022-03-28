"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"sequence_main.py"
Predictive processes at the single neuron level 

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
                    Single neuron trained on a predictive coding scheme
                    """
                    )

'options'
parser.add_argument('--online',type=str,default='False',
                    help='train mode with online approxi algorithm')
'training algorithm'
parser.add_argument('--l_rate',type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs')
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--batch', type=int, default=64,
                    help='number of batches')   
'architecture'
parser.add_argument('--N', type=int, default=40) 
parser.add_argument('--T', type=int, default=40) 
parser.add_argument('--timing', type=int, default=40) 
'neuronal model'
parser.add_argument('--dt', type=float, default= 1) 
parser.add_argument('--tau_m', type=float, default= 2000.) 
parser.add_argument('--v_th', type=float, default= .6)
parser.add_argument('--dtype', type=str, default=torch.float) 

par = parser.parse_args()

'additional parameters'
par.device = "cuda" if torch.cuda.is_available() else "cpu"
'input'
par.tau_x = 2.
par.freq = 0.
par.jitt = 2.

print("""
      OPTIMIZATION on EPISODIC MEMORY TASK with LIF\t
      eprop: {} ; n {}
      eprop approximation: {}
      tau_m: {} ms
      iteration {}
      """.format(par.eprop,par.n,par.eprop_approx,par.tau_m,par.rep))

'train'
accuracy, runtime, loss_class,loss_reg =  train.train(par)

'-----------'

<<<<<<< HEAD
print('saving')
np.save('data/accuracy_tau_{}_eprop_{}_approx_{}_n_{}_rep_{}'.format(par.tau_m,par.eprop,par.eprop_approx,par.n,par.rep),accuracy)
np.save('data/runtime_tau_{}_eprop_{}_approx_{}_n_{}_rep_{}'.format(par.tau_m,par.eprop,par.eprop_approx,par.n,par.rep),runtime)     
np.save('data/loss_class_tau_{}_eprop_{}_approx_{}_n_{}_rep_{}'.format(par.tau_m,par.eprop,par.eprop_approx,par.n,par.rep),loss_class)  
np.save('data/loss_reg_tau_{}_eprop_{}_approx_{}_n_{}_rep_{}'.format(par.tau_m,par.eprop,par.eprop_approx,par.n,par.rep),loss_reg)
=======
from predictive_neuron import models 
>>>>>>> f62acb932a3863c2168685ec0e08322a28f88b12

