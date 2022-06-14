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
import os
import torch
import torch.nn.functional as F
import torchvision

import matplotlib.colors as colors

'--------------'

def get_firing_rate(par,x,Dt=1):
    
    bins, bin = int(par.T*par.dt/Dt), int(Dt/par.dt)
    fr = []
    for k in range(bins):
        fr.append((torch.sum(x[:,k*bin:(k+1)*bin])*(1e3/Dt))/par.N)
    
    return np.array(fr)

def get_density(par,x):
    
    bins = np.arange(par.T).tolist()
    step = int(par.tau_m/par.dt)
    bins = [bins[i:i+step] for i in range(0,len(bins),int(1/par.dt))]
    density = [torch.sum(x[0,bins[k],:]).item() for k in range(len(bins))]

    return density

'--------------'

def get_sequence(par,timing):
    
    if par.offset == 'True': timing += np.random.randint(0,par.T/2)
        
    if par.fr_noise == 'True':
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        
    if par.jitter_noise == 'True':
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N))/par.dt
             x_data[b,timing_err.tolist(),range(par.N)] = 1
    else:
        x_data[:,timing,range(par.N)] = 1
             
    density = get_density(par,x_data)
    fr = get_firing_rate(par,x_data)
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1), density, fr

def get_sequence_density(par,timing,mu=None,jitter=None,offset=None):
    
    if offset: timing += np.random.randint(0,par.T/2)
        
    if par.fr_noise == 'True':
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        
    if par.jitter_noise == 'True':
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N))/par.dt
             x_data[b,timing_err.tolist(),range(par.N)] = 1
    else:
        x_data[:,timing,range(par.N)] = 1
             
    density = get_density(par,x_data)
    fr = get_firing_rate(par,x_data)
    
    return density, fr

def sequence_capacity(par,timing):
    
    if par.fr_noise == 'True':
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    
    'create sequence' 
    for b in range(par.batch):
        if par.jitter_noise == 'True':
            timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N_sub))/par.dt
            x_data[b,timing_err,b*par.N_sub + np.arange(par.N_sub)] = 1
        else: x_data[b,timing,b*par.N_sub + np.arange(par.N_sub)] = 1
        
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)

def get_sequence_stdp(par,timing):
    
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    x_data[:,timing,range(par.N_stdp)]= 1
    density = get_density(par,x_data)
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

'--------------'

def get_sequence_NumPy(par,timing):
    
    x_data = np.zeros((par.N,par.T))
    x_data[range(par.N),timing.astype(int)] = 1
    for k in range(par.N):
        x_data[k,:] = np.convolve(x_data[k,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_m))[:par.T]   
        
    return x_data

'------------'

def get_pattern(par):
    
    if par.offset == 'True': offset = np.random.randint(0,par.T/2)
    else: offset = 0
    
    if par.fr_noise == 'True':
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    
    prob = par.freq_pattern*par.dt
    x_data[:,offset:offset+par.T_pattern,:] = 0
    
    if par.jitter_noise == 'True':
        for b in range(par.batch):
            for n in range(par.N):
                timing_err = np.array(par.timing[n]) + np.random.randint(-par.jitter,par.jitter,len(par.timing[n]))/par.dt
                x_data[b,timing_err,n] = 1
    else:
        x_data[:,offset:offset+par.T_pattern,:][par.mask<prob] = 1
    
    density = get_density(par,x_data)
    fr = get_firing_rate(par,x_data)
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1), density, fr
    
def get_pattern_density(par,mu=None,offset=None):
        
    'add background firing noise'
    if mu:
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    
    prob = par.freq_pattern*par.dt
    x_data[:,offset:offset+par.T_pattern,:] = 0
    x_data[:,offset:offset+par.T_pattern,:][par.mask<prob] = 1
    
    density = get_density(par,x_data)
    fr = get_firing_rate(par,x_data)
    
    return density, fr
    

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

def get_pattern_fixed_noise(par,avg=None):
    
    prob = par.freq*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1
#    for k in range(par.N):
#        idx = torch.where(x_data[0,:,k]==1)[0]
#        if len(idx)>0: x_data[0,idx[np.random.randint(0,len(idx))],k] = 0
    
    x_data[par.mask<par.freq_pattern*par.dt] = 1
    x_data[:,par.timing,range(par.N)] = 1
        
    density = get_density(par,x_data)
    fr = get_firing_rate(par,x_data)
    
    if avg: return density, fr
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density, fr

'------------'

def get_multi_sequence(par,timing):
    
    'create sequence'    
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    for k in range(par.sequences):
        
        x_data[:,timing[k],par.N_sequences[k]] = 1

    'compute pattern density'
    density = get_density(par,x_data)
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

def get_multi_sequence_noise(par,timing):
    
    'create sequence'    
    prob = par.freq*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1
    
    for k in range(par.sequences):
        
        x_data[:,timing[k],par.N_sequences[k]] = 1

    'compute pattern density'
    density = get_density(par,x_data)
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

def get_multi_pattern_fixed(par):
    
    'create pattern'    
    probs = par.freqs*par.dt
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    for k in range(par.patterns):
        mask = torch.rand(par.batch,par.T_patterns[k],par.N_patterns[k]).to(par.device)
        x_data[:,k*(par.T_patterns[k]+par.DT):(k+1)*(par.T_patterns[k])+k*par.DT,k*par.N_patterns[k]:(k+1)*par.N_patterns[k]][mask<probs[k]] = 1

        timing = np.random.randint(k*(par.T_patterns[k]+par.DT),(k+1)*(par.T_patterns[k])+k*par.DT,size=par.N_patterns[k])
        x_data[:,timing,np.arange(k*par.N_patterns[k],(k+1)*par.N_patterns[k])] = 1
    
    'compute pattern density'
    density = get_pattern_density(par,x_data)
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

'--------------------'
'--------------------'

def current2spktime(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))
    T[idx] = tmax
    return T

def sparse_from_fashionMNIST(par,x_data,y_data,shuffle=True):
    """ 
    this generator takes a spike dataset and generates spiking network input as sparse tensors. 
    args:
        x_data: ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y_data: labels
    """
    
    'get labels and batch size'
    labels_ = np.array(y_data,dtype=np.int)
    number_of_batches = len(x_data)//par.batch
    sample_index = np.arange(len(x_data))
    
    'compute discrete spike times'
    tau_eff = 20/par.dt
    spk_times = np.array(current2spktime(x_data,tau=tau_eff,tmax=par.T), dtype=np.int)
    unit_numbers = np.arange(par.n_in)

    if shuffle:
        np.random.shuffle(sample_index)
    
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[par.batch*counter:par.batch*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            
            c = spk_times[idx]<par.T
            times, units = spk_times[idx][c], unit_numbers[c]
            
            batch = [bc for _ in range(len(times))]
            
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(par.device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(par.device)
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([par.batch,par.T,par.n_in])).to(par.device)
        y_batch = torch.tensor(labels_[batch_index],device=par.device)

        yield X_batch.to(device=par.device), y_batch.to(device=par.device),

        counter += 1

def get_fashionMNIST(par):
    
    root = os.path.expanduser("~/data/datasets/torch/fashion-mnist")
    train_dataset = torchvision.datasets.FashionMNIST(root, train=True, 
                                                      transform=None, 
                                                      target_transform=None, 
                                                      download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root, train=False, 
                                                     transform=None, 
                                                     target_transform=None, 
                                                     download=True)
    'standardize data'
    x_train = torch.tensor(train_dataset.train_data, device=par.device, dtype=par.dtype)
    x_train = x_train.reshape(x_train.shape[0],-1)/255
    x_test = torch.tensor(test_dataset.test_data, device=par.device, dtype=par.dtype)
    x_test = x_test.reshape(x_test.shape[0],-1)/255
    
    y_train = torch.tensor(train_dataset.train_labels, device=par.device, dtype=par.dtype)
    y_test  = torch.tensor(test_dataset.test_labels, device=par.device, dtype=par.dtype)
        
    return x_train, y_train, x_test, y_test

'--------------------'
'--------------------'

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