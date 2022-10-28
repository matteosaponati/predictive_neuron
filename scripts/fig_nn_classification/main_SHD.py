"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-

----------------------------------------------
"main_SHD.py"
- add description -
----------------------------------------------
Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import numpy as np
import argparse
import os
import json

import train

parser = argparse.ArgumentParser(
                    description="""
                    shallow hidden layer on classification task, SHD dataset
                    """
                    )

parser.add_argument('--path',type=str,default='/mnt/pns/departmentN4/snn_neuroscience_methods/SHD_dataset/',
                    help='directory to save data')
'training algorithm'
parser.add_argument('--optimizer',type=str, 
                    choices=['SGD','NAG','Adam'],default='Adam',
                    help='choice of optimizer')
parser.add_argument('--l_rate',type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=250)
parser.add_argument('--seed', type=int, default=1992)
parser.add_argument('--N', type=int, default=256,
                    help='number of batches')

parser.add_argument('--reg_fr', type=int, default=10,
                    help='target pop firing rate [Hz]')
parser.add_argument('--reg_coeff', type=int, default=1)

parser.add_argument('--rep',type=int,default=1,
                    help='iteration number')
parser.add_argument('--save_output',type=str,default='False',
                    help='set output save')

'architecture'
parser.add_argument('--n', type=int, default=256) 
parser.add_argument('--n_out', type=int, default=20)
parser.add_argument('--data_augment',type=str,default='False',
                    help='set data augmentation')
parser.add_argument('--n_in', type=int, default=700) 
parser.add_argument('--T', type=int, default=1000) 

'neuronal model'
parser.add_argument('--dt', type=float, default= 1) 
parser.add_argument('--tau_m', type=float, default= 20.) 
parser.add_argument('--v_th', type=float, default= 1.)
parser.add_argument('--dtype', type=str, default=torch.float) 

par = parser.parse_args()

'additional parameters'
par.device = "cuda" if torch.cuda.is_available() else "cpu"
par.alpha = float(np.exp(-par.dt/par.tau_m))
par.prompt = 10

print("""
      CLASSIFICATION with LIF network\t
      
      optimizer: {}
      l_rate: {}, n: {}
      tau_m: {} ms
      surr gradient scale: {}
      
      data augmentation {}
      T: {} ; binning: {}
      n_in: {} ; binning {}
      
      iteration {}
      
      """.format(par.optimizer,par.l_rate,par.n,par.tau_m,40,par.data_augment,
                  par.T,round(1.4e3/par.T,2),par.n_in,round(700/par.n_in,2),
                  par.rep))

'train'
accuracy, runtime, loss_class, loss_reg, accuracy_test = train.train(par)

'-----------'
'save'
par.dtype = 'torch.float'
par.savedir = os.path.join(par.path,'T_{}_nin_{}_n_{}_tau_{}'.format(
                                            par.T,par.n_in,par.n,par.tau_m))
if not os.path.exists(par.savedir): os.makedirs(par.savedir)
   
with open(os.path.join(par.savedir,'par.txt'),'w') as f:
    json.dump(par.__dict__,f,indent=2)

np.save(os.path.join(par.savedir,'accuracy_rep_{}'.format(par.rep)),
                    accuracy)
np.save(os.path.join(par.savedir,'loss_class_rep_{}'.format(par.rep)),
                    loss_class)
np.save(os.path.join(par.savedir,'loss_reg_rep_{}'.format(par.rep)),
                    loss_reg)
np.save(os.path.join(par.savedir,'runtime_rep_{}'.format(par.rep)),
                    runtime)
