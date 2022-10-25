"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"funs.py"
auxiliary functions to create input data and plots

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.colors as colors

'---------------------------------------------'
'get sequence - PyTorch version'

def get_firing_rate(par,x,Dt=1):
    
    bins, bin = int(par.T*par.dt/Dt), int(Dt/par.dt)
    fr = []
    for k in range(bins):
        fr.append((torch.sum(x[:,k*bin:(k+1)*bin])*(1e3/Dt))/par.N)
    
    return np.array(fr)

def get_sequence(par,timing,onset=None):
    
    if par.onset == True: timing += onset
        
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N)*par.dt)/1000
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        for n in range(par.N): x_data[:,:,n][torch.rand(par.batch,par.T).to(par.device)<prob[n]] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        
    if par.jitter_noise == True:
         for b in range(par.batch):
             
             timing_err = np.array(timing) + \
                             np.random.randint(-par.jitter,par.jitter,len(timing))/par.dt
             x_data[b,timing_err.tolist(),range(len(timing))] = 1
    else:
        x_data[:,timing,range(len(timing))] = 1
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1), onset

def get_multisequence(par,timing):
    
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N)*par.dt)/1000
        # setting all background firing to the same Poisson firing rate
        # prob = ((par.freq*par.dt)/1000)*torch.ones(par.N)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        for n in range(par.N): x_data[:,:,n][torch.rand(par.batch,par.T).to(par.device)<prob[n]] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    
    'create sequence' 
    for b in range(par.batch):
        if par.jitter_noise == True:
            timing_err = np.array(timing[b]) + \
                            (np.random.randint(-par.jitter,par.jitter,par.N_sub))/par.dt
            x_data[b,timing_err,par.N_subseq[b]] = 1
        else: x_data[b,timing[b],par.N_subseq[b]] = 1
        
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                          padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)

def get_sequence_rhythms(par,timing,onset=None):
    
    if par.onset == True: timing += onset
        
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N_dist)*par.dt)/1000
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        for n in range(par.N_seq,par.N): x_data[:,:,n][torch.rand(par.batch,par.T).to(par.device)<prob[n]] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        
    if par.jitter_noise == True:
         for b in range(par.batch):
             for n in range(par.N_seq):
             
                 timing_err = np.array(timing[n]) + \
                                 np.random.randint(-par.jitter,par.jitter,len(timing))/par.dt
                 x_data[b,timing_err.tolist(),n] = 1
    else:
        for n in range(par.N_seq):
            x_data[:,timing[n],n] = 1
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1), onset

'--------------'
'get sequence - NumPy version + get sequence for STDP protocols'

def get_sequence_NumPy(par,timing,onset):
    
    'set random input onset'
    if par.onset == True: timing = timing.copy() + onset
    
    'set background firing'
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N)*par.dt)/1000
        x_data = np.zeros((par.N,par.T))
        for n in range(par.N): x_data[n,:][np.random.rand(par.T)<prob[n]] = 1        
    else: x_data = np.zeros((par.N,par.T))

    'set jitter'        
    if par.jitter_noise == True:
        timing_err = np.array(timing) \
                        + np.random.randint(-par.jitter,par.jitter,len(timing))/par.dt
        x_data[range(par.N_seq),timing_err.astype(int).tolist()] = 1
    else: x_data[range(par.N_seq),timing] = 1
        
    'synaptic time constant'
    for n in range(par.N):
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]      
        
    return x_data

def get_multisequence_NumPy(par,timing):
    
    'set background firing'
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N)*par.dt)/1000
        x_data = np.zeros((par.N,par.T))
        for n in range(par.N): x_data[n,:][np.random.rand(par.T)<prob[n]] = 1        
    else: x_data = np.zeros((par.N,par.T))

    'set jitter' 
    for b in range(par.batch):
        if par.jitter_noise == True:
            timing_err = timing[b]  \
                           + (np.random.randint(-par.jitter,par.jitter,par.N_sub))/par.dt
            x_data[par.N_subseq[b].astype(int),(timing_err).astype(int)] = 1
        else: x_data[par.N_subseq[b],(timing_err).astype(int)] = 1
        
    'synaptic time constant'
    for n in range(par.N):
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]      
        
    return x_data

def get_rhythms_NumPy(par,timing,onset=None):
    
    'set random input onset'
    if par.onset == True: timing = timing.copy() + onset
    
    'set background firing'
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N_dist)*par.dt)/1000
        x_data = np.zeros((par.N,par.T))
        for n in range(par.N_dist): x_data[par.N_seq+n,:][np.random.rand(par.T)<prob[n]] = 1        
    else: x_data = np.zeros((par.N,par.T))

    'set jitter'        
    if par.jitter_noise == True:
        for n in range(par.N_seq):
            
            if np.random.rand() < par.cycle_prob:
                timing_err = np.array(timing[n][0]) + \
                            np.random.randint(-par.jitter,par.jitter,len(timing[n][0]))/par.dt
                x_data[n,timing_err.astype(int).tolist()] = 1
    else: 
        for n in range(par.N_seq): x_data[n,timing] = 1
        
    'synaptic time constant'
    for n in range(par.N):
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]      
        
    return x_data

def get_sequence_stdp(par,timing):
    
    x_data = np.zeros((par.N,par.T))
    
    for n in range(par.N):
        x_data[n,timing[n]]= 1
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]      
        
    return x_data

'---------------------------------------------'
'get sequence neural network with trainable recurrent connections - PyTorch version'

def get_sequence_nn_selforg(par,timing):
    
    'loop on neurons in the network'
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.freq_noise == True:
            prob = (np.random.randint(0,par.freq,par.n_in)*par.dt)/1000
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            for nin in range(par.n_in): x[:,:,nin][torch.rand(par.batch,par.T).to(par.device)<prob[nin]] = 1        
        else:
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            
        'create sequence + jitter' 
        for b in range(par.batch):
            if par.jitter_noise == True:
                timing_err = np.array(timing[n][b]) \
                              +  np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt
                x[b,timing_err,range(par.n_in)] = 1
            else: x[b,timing[n][b],range(par.n_in)] = 1
        
        'filtering'
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
            
        'add to total input'
        x_data.append(x.permute(0,2,1))

    return torch.stack(x_data,dim=3)

'--------------'
'get sequence neural network with trainable recurrent connections - NumPy version'

def get_sequence_nn_selforg_NumPy(par,timing):
    
    'loop on neurons in the network'
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.freq_noise == True:
            prob = (np.random.randint(0,par.freq,par.n_in)*par.dt)/1000
            x = np.zeros((par.n_in,par.T))
            for nin in range(par.n_in): x[nin,:][np.random.rand(par.T)<prob[nin]] = 1        
        else:
            x = np.zeros((par.n_in,par.T))
            
        'create sequence + jitter'
        if par.jitter_noise == True:
            timing_err = np.array(timing[n]) \
                          +  (np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt).astype(int)
            x[range(par.n_in),timing_err] = 1
        else: x[range(par.n_in),timing[n]] = 1
        
        'synaptic time constant'
        for nin in range(par.n_in):
            x[nin,:] = np.convolve(x[nin,:],
                          np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]   
            
        'add to total input'
        x_data.append(x)

    return np.stack(x_data,axis=2)

'---------------------------------------------'
'get sequences neural network with inhibition - PyTorch version'

def get_nn_multisequence(par,timing):
    
    x_data  = []
    for n in range(par.nn):
        
        'add background firing'         
        if par.freq_noise == True:
            prob = (np.random.randint(0,par.freq,par.n_in)*par.dt)/1000
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)     
            for nin in range(par.n_in): x[:,:,nin][torch.rand(par.batch,par.T).to(par.device)<prob[nin]] = 1 
        else:
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            
        'create sequence + jitter' 
        for b in range(par.batch):
            if par.jitter_noise == True:
                timing_err = np.array(timing[n][b]) \
                              +  np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt
                x[b,timing_err,range(par.n_in)] = 1
            else: x[b,timing[n][b],range(par.n_in)] = 1
        
        'filtering'
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
            
        'add to total input'
        x_data.append(x.permute(0,2,1))

    return torch.stack(x_data,dim=3)

'---------------------------------------------'

"auxiliary functions plots"

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
