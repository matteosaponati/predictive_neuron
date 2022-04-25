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
    
    'create sequence'
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    x_data[:,timing,range(par.N)] = 1
    density = get_pattern_density(par,x_data)
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density
    
def get_sequence_noise(par,timing,mu=None,jitter=None):
        
    'add background firing noise'
    if mu:
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1
        x_data[:,timing,range(par.N)] = 1
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[:,timing,range(par.N)] = 1
    'add jitter in sequence'
    if jitter:
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N))/par.dt
             x_data[b,timing_err.tolist(),range(par.N)] = 1
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1)

def get_multi_sequence(par,timing):
    
    'create sequence'    
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    for k in range(par.sequences):
        
        x_data[:,timing[k],par.N_sequences[k]] = 1

    'compute pattern density'
    density = get_pattern_density(par,x_data)
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

'------------'

def get_pattern_density(par,x):
    
    bins = np.arange(par.T).tolist()
    step = int(par.tau_m/par.dt)
    bins = [bins[i:i+step] for i in range(0,len(bins),int(1/par.dt))]
    density = [torch.sum(x[0,bins[k],:]).item() for k in range(len(bins))]

    return density

def get_pattern(par):
    
    'create pattern'    
    prob = par.freq_pattern*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1
    'compute pattern density'
    density = get_pattern_density(par,x_data)
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

def get_pattern_fixed(par):
    
    prob = par.freq_pattern*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1
    
    timing = np.random.randint(0,par.T,size=par.N)
    x_data[:,timing,range(par.N)] = 1
    
    'compute pattern density'
    density = get_pattern_density(par,x_data)
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

def get_pattern_noise(par,timing,mu=None,jitter=None):
    
    x_data = get_pattern(par)
    'add background firing noise'
    if mu:
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1
    'add jitter in sequence'
    if jitter:
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N))/par.dt
             x_data[b,timing_err.tolist(),range(par.N)] = 1
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1)

def get_multi_pattern(par):
    
    'create pattern'    
    probs = par.freqs*par.dt
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    for k in range(par.patterns):
        mask = torch.rand(par.batch,par.T_patterns[k],par.N_patterns[k]).to(par.device)
        x_data[:,k*(par.T_patterns[k]+par.DT):(k+1)*(par.T_patterns[k])+k*par.DT,k*par.N_patterns[k]:(k+1)*par.N_patterns[k]][mask<probs[k]] = 1

    'compute pattern density'
    density = get_pattern_density(par,x_data)
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

#'------------'
#
#def get_sequence_capacity(par,timing):
#    
#    'define the set of sequences'
#    seq = [[a,b] for a in range(par.N) for b in range(par.N) if a!=b]
#    
#    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
#    for k in range(par.batch):
#        x_data[k,timing,seq[k]] = 1
#    
#    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
#                                for i in range(par.T)]).view(1,1,-1).float()
#    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
#                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
#    
#    return x_data.permute(0,2,1)