"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"funs.py"
auxiliary functions to create input data

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F

'--------------'

def get_sequence(par,timing):

    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    x_data[:,timing,range(par.N)] = 1

    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float()
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)
    
def get_sequence_noise(par,timing,mu=None,jitter=None):
        
    'background firing noise'
    if mu:
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1
        x_data[:,timing,range(par.N)] = 1
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[:,timing,range(par.N)] = 1
    
    'jitter in sequence'
    if jitter:
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N))/par.dt
             x_data[b,timing_err.tolist(),range(par.N)] = 1
            
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float()
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1)

'------------'

#def pattern(par):
#
#    prob = par.freq*par.dt
#    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
#  
#    return mask, prob
#
#def get_pattern(par,mask,prob):
#    
#    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
#    
#    
#    
#    prob = par.freq*par.dt
#        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
#        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
#        x_data[mask<prob] = 1
#        x_data[:,timing,range(par.N)] = 1
#    
#    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
#    x_data[:,timing,range(par.N)] = 1
#
#    
#    return

def get_sequence_capacity(par,timing):
    
    'define the set of sequences'
    seq = [[a,b] for a in range(par.N) for b in range(par.N) if a!=b]
    
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    for k in range(par.batch):
        x_data[k,timing,seq[k]] = 1
    
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float()
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1)