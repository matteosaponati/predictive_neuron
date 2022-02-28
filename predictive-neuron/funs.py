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

import numpy as np
import torch
import torch.nn as nn

'create input pattern'
def sequence(par,timing):    

    inputs = np.zeros((par.N,par.T))
    for k in range(par.N):
        inputs[k,np.array(timing[k]/par.dt).astype('int')]= 1
        inputs[k,:] = np.convolve(inputs[k,:],np.exp(-np.arange(0,par.T,par.dt)/par.tau))[:par.T]        
    return inputs