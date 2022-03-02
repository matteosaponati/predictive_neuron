"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"models.py"
Predictive processes at the single neuron level 

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn as nn

'create input pattern'

def get_sequence(par,timing):

    prob = par.freq*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1.0
    
    for k in range(par.N):
        x_data[:,k,timing[k]] = 1
        
    ## convolve with kernel
    
    return x_data