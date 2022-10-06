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

'--------------'

def get_firing_rate(par,x,Dt=1):
    
    bins, bin = int(par.T*par.dt/Dt), int(Dt/par.dt)
    fr = []
    for k in range(bins):
        fr.append((torch.sum(x[:,k*bin:(k+1)*bin])*(1e3/Dt))/par.N)
    
    return np.array(fr)
#
#def get_density(par,x):
#    
#    bins = np.arange(par.T).tolist()
#    step = int(par.tau_m/par.dt)
#    bins = [bins[i:i+step] for i in range(0,len(bins),int(1/par.dt))]
#    density = [torch.sum(x[0,bins[k],:]).item() for k in range(len(bins))]
#
#    return density

'--------------'

def get_sequence(par,timing):
    
    if par.offset == True: timing += np.random.randint(0,par.T/2)
        
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N)*par.dt)/1000
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        for n in range(par.N): x_data[:,:,n][torch.rand(par.batch,par.T).to(par.device)<prob[n]] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        
    if par.jitter_noise == True:
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,len(timing)))/par.dt
             x_data[b,timing_err.tolist(),range(len(timing))] = 1
    else:
        x_data[:,timing,range(len(timing))] = 1
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1)

def get_sequence_fr(par,timing):
    
    if par.offset == True: timing += np.random.randint(0,par.T/2)
        
    if par.freq_noise == True:
        prob = (np.random.randint(0,par.freq,par.N)*par.dt)/1000
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        for n in range(par.N): x_data[:,:,n][torch.rand(par.batch,par.T).to(par.device)<prob[n]] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        
    if par.jitter_noise == True:
         for b in range(par.batch):
             timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,len(timing)))/par.dt
             x_data[b,timing_err.tolist(),range(len(timing))] = 1
    else:
        x_data[:,timing,range(len(timing))] = 1
             
#    density = get_density(par,x_data)
    fr = get_firing_rate(par,x_data)
    
    return fr

def sequence_capacity(par,timing):
    
    if par.fr_noise == True:
        prob = par.freq*par.dt
        mask = torch.rand(par.batch,par.T,par.N).to(par.device)
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
        x_data[mask<prob] = 1        
    else:
        x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    
    'create sequence' 
    for b in range(par.batch):
        if par.jitter_noise == True:
            timing_err = np.array(timing) + (np.random.randint(-par.jitter,par.jitter,par.N_sub))/par.dt
            x_data[b,timing_err,b*par.N_sub + np.arange(par.N_sub)] = 1
        else: x_data[b,timing,b*par.N_sub + np.arange(par.N_sub)] = 1
        
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)

'--------------'

'get sequence - NumPy version'
def get_sequence_NumPy(par,timing):
    
    x_data = np.zeros((par.N,par.T))
    
    for n in range(par.N):
        x_data[n,timing[n]]= 1
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]      
        
    return x_data

'numerical solution and training for sequences - NumPy version'
def train_NumPy(par,neuron,x_data):
    w1, w2 = [], []
    v_tot, spk_tot = [],[]
    for e in range(par.epochs):        
        neuron.state()
        neuron, v, spk = forward(par,neuron,x_data)    
        v_tot.append(v)
        spk_tot.append(spk)
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%100 == 0: print(e)        
    return w1, w2, v_tot, spk_tot

'--------------'

'pre-synaptic inputs for STDP - NumPy version'
def get_sequence_stdp(par,timing):
    
    x_data = np.zeros((par.N,par.T))
    
    for n in range(par.N):
        x_data[n,timing[n]]= 1
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]      
        
    return x_data

'numerical solution and training for STDP - NumPy version'
def forward(par,neuron,x_data):    
    v,z = [], []
    for t in range(par.T):    
        v.append(neuron.v) 
        neuron(x_data[:,t])          
        if neuron.z != 0: z.append(t*par.dt)    
    return neuron, v, z
def train(par,neuron,x_data):
    w1, w2 = [], []
    for e in range(par.epochs):        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)        
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        if e%10 == 0: print(e)        
    return w1, w2
'---------------------------------------------'

'--------------'

def get_sequence_nn_selforg(par):
    
    'create timing'
    if par.random==True:
        timing = [[] for n in range(par.nn)]
        for n in range(par.nn):
            for b in range(par.batch): 
                spk_times = np.random.randint(0,(par.Dt/par.dt)*par.n_in,size=par.n_in)
                timing[n].append(spk_times+n*(par.n_in*par.Dt/par.dt)+ par.delay/par.dt)
    else: 
        timing = [[] for n in range(par.nn)]
        spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
        for n in range(par.nn):
            for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt) #*(par.n_in*par.Dt/par.dt)+ 
            
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.freq_noise == True:
            prob = par.freq*par.dt
            mask = torch.rand(par.batch,par.T,par.n_in).to(par.device)
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            x[mask<prob] = 1        
        else:
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            
        'create sequence + jitter' 
        for b in range(par.batch):
            if par.jitter_noise == True:
                timing_err = np.array(timing[n][b]) + np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt
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

'------------'

def get_multisequence_nn(par,timing):
    
    x_data  = []
    for n in range(par.nn):
        
        'add background firing'         
        if par.freq_noise == True:
            prob = par.freq*par.dt
            mask = torch.rand(par.batch,par.T,par.n_in).to(par.device)
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            x[mask<prob] = 1        
        else:
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            
        'create sequence + jitter' 
        for b in range(par.batch):
            if par.jitter_noise == True:
                timing_err = np.array(timing[n][b]) + np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt
                x[b,timing_err,range(par.n_in)] = 1
            else: x[b,timing[n][b],range(par.n_in)] = 1
        
        'filtering'
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
        x_data.append(x.permute(0,2,1))

    return torch.stack(x_data,dim=3)

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
