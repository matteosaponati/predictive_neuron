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

def get_sequence_stdp(par,timing):
    
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    for k in range(par.N):
        x_data[:,timing[k],k]= 1
    density = get_pattern_density(par,x_data)
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1), density

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
    number_of_batches = len(x_data)//par.N
    sample_index = np.arange(len(x_data))
    
    'compute discrete spike times'
    tau_eff = 20/par.dt
    spk_times = np.array(current2spktime(x_data,tau=tau_eff,tmax=par.T), dtype=np.int)
    unit_numbers = np.arange(par.n_in)

    if shuffle:
        np.random.shuffle(sample_index)
    
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[par.N*counter:par.N*(counter+1)]

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
    
        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([par.N,par.T,par.n_in])).to(par.device)
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