"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"funs_train_inhibition.py"
auxiliary functions to train the network model with recurrent inhibition

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn

from predictive_neuron import funs, models

'---------------------------------------------------------------------------'
'numerical solution and training for sequences - PyTorch version'

def initialize_nn(par):
    """
    initialize the network as a list of NeuronClass_nn objects.
    each NeuronClass_nn instance has independent forward and backward pass
    the weights of each NeuronClass_nn instance are initialized according to par
    """
    
    network = []    
    for n in range(par.nn):
        'initialize neuron'
        network.append(models.NeuronClass_nn(par))
        'initialize input weights'
        if par.init == 'trunc_gauss':
            network[n].w = nn.Parameter(torch.empty(par.N)).to(par.device)
            torch.nn.init.trunc_normal_(network[n].w, mean=par.init_mean, std=1/np.sqrt(par.N),
                                        a=par.init_a,b=par.init_b) 
        if par.init == 'uniform':
            network[n].w = nn.Parameter(torch.FloatTensor(par.N).uniform_(0.,par.init_mean))
        if par.init == 'fixed':
            network[n].w = nn.Parameter(par.init_mean*torch.ones(par.N))
        'initialize recurrent weights'    
        network[n].wrec = par.w0_rec*torch.ones(par.nn-1)
    
    return network

def forward_nn(par,network,x):
    """
    set list of membrane voltage and output spike variables for each neuron
    update the state variables at each timestep t and save values in lists
    """
    
    v = [[] for n in range(par.nn)]
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    'forward pass across time'
    for t in range(par.T):            
        'get voltage at timestep t-1'
        for n in range(par.nn):
            v[n].append(network[n].v)  
        'get recurrent inputs'
        z_out = []
        for n in range(par.nn):
            z_out.append(torch.stack([network[n].z_out.detach()
                                      for k in range(par.nn)
                                      if k != n],dim=1))
        for n in range(par.nn):
            'update neuron states - forward pass'
            network[n](x[n][:,t],z_out[n])
            'get output spike at timestep t'
            for b in range(par.batch):
                if network[n].z[b].item() != 0: z[n][b].append(t*par.dt)    
    
    return network, [torch.stack(v[n],dim=1) for n in range(par.nn)], z

def train_nn(par,network,x=None,timing=None):
    """
    define the MSE loss function
    
    optimization step with independent neurons:
    - set a list of optimizers, one for each neuron in the network
    - compute forward pass for each neuron. The recurrent interactions are not
    considered in the computational graph
    - compute the MSE loss and run optimizer.step() for each neuron separately 
    
    save relevant variables
    """
    
    'define loss function and optimizers'
    loss = nn.MSELoss(reduction='sum')
    optimizerList = []
    for n in range(par.nn):
        if par.optimizer == 'Adam':
            optimizerList.append(torch.optim.Adam(network[n].parameters(),
                                  lr=par.eta,betas=(.9,.999)))
        elif par.optimizer == 'SGD':
            optimizerList.append(torch.optim.SGD(network[n].parameters(),lr=par.eta))

    'allocate outputs'
    loss_list, w_list = [], []
    v_list, spk_list = [], []        

    'train network across epochs'
    for e in range(par.epochs):
        
        'load or create data'
        if par.noise == True:
            if par.upload_data == 1:
                x = torch.load(par.load_dir+'x_data_{}.npy'.format(1))
            else: x = funs.get_multisequence_nn(par,timing)
            
        'initialize neuron state and solve dynamics (forward pass)'
        for n in range(par.nn): 
            network[n].state()
        network, v, z = forward_nn(par,network,x)
        
        'optimization step'
        EList, x_hatList = [], []
        for n in range(par.nn):
            x_hatList.append(torch.einsum("bt,j->btj",v[n],network[n].w))
            EList.append(.5*loss(x_hatList[n],x[n]))
        for n in range(par.nn):
            optimizerList[n].zero_grad()
        for n in range(par.nn):
            EList[n].backward()
        for n in range(par.nn):
            optimizerList[n].step()
        
        'save output'
        v_list.append(v)
        spk_list.append(z)
        w_list.append([network[n].w.detach().clone().numpy() for n in range(par.nn)])
       
        if e%50 == 0: 
            print('epoch {} out of {}'.format(e,par.epochs))
    
    return w_list, v_list, spk_list, loss_list

'---------------------------------------------------------------------------'
'numerical solution and training for sequences - NumPy version'

def initialize_nn_NumPy(par):
    """
    initialize the network as a list of NeuronClass_nn objects.
    each NeuronClass_nn instance has independent forward-backward pass
    the weights of each NeuronClass_nn instance are initialized according to par
    """
    
    network = []    
    for n in range(par.nn):
        'initialize neuron'
        network.append(models.NeuronClass_nn_NumPy(par))
        'initialize input weights'
        if par.init == 'trunc_gauss':
            network[n].w = stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.N)), 
                              (par.init_b-par.init_mean)/(1/np.sqrt(par.N)), 
                              loc=par.init_mean, scale=1/np.sqrt(par.N)).rvs(par.N)
        if par.init == 'uniform':
            network[n].w = np.random.uniform(0,par.init_mean,par.N)
        if par.init == 'fixed':
            network[n].w = par.init_mean*np.ones(par.N)
        'initialize recurrent weights'    
        network[n].wrec = par.w0_rec*np.ones(par.nn-1)
    
    return network

def forward_nn_NumPy(par,network,x):
    """
    set list of membrane voltage and output spike variables for each neuron
    update the state variables at each timestep t and save values in lists
    """
    
#    v = [[] for n in range(par.nn)]
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    'forward pass across time'
    for t in range(par.T):            
        'get voltage at timestep t-1'
#        for n in range(par.nn):
#            v[n].append(network[n].v)  
        for n in range(par.nn):
            'update neuron states - forward pass'
            network[n](x[n][:,t],np.stack([network[n].z_out 
                                   for k in range(par.nn)
                                      if k != n],axis=1))
            'get output spike at timestep t'
            for b in range(par.batch):
                if network[n].z[b].item() != 0: z[n][b].append(t*par.dt)    
    
#    return network, [np.stack(v[n],axis=1) for n in range(par.nn)], z
    return network, z

def train_nn_NumPy(par,network,x=None,timing=None):
    """
    optimization step with independent neurons:
    - compute forward-backward pass for each neuron. The recurrent interactions are not
    considered in the computational graph
    - save relevant variables
    """

    'allocate outputs'
    loss_list, w_list = [], []
#    v_list, spk_list = [], []       
    spk_list = []

    'train network across epochs'
    for e in range(par.epochs):
        
        'load or create data'
        if par.noise == True:
            if par.upload_data == 1:
                x = np.load(par.load_dir+'x_data_{}.npy'.format(1))
            else: x = funs.get_multisequence_nn_NumPy(par,timing)
            
        'initialize neuron state and solve dynamics (forward+backward pass)'
        for n in range(par.nn): 
            network[n].state()
#        network, v, z = forward_nn_NumPy(par,network,x)
        network, z = forward_nn_NumPy(par,network,x)
                
        'save output'
#        v_list.append(v)
        spk_list.append(z)
        w_list.append([network[n].w for n in range(par.nn)])
       
        if e%50 == 0: 
            print('epoch {} out of {}'.format(e,par.epochs))
    
#    return w_list, v_list, spk_list, loss_list
    return w_list, spk_list, loss_list

'--------'

def initialize_weights_nn_PyTorch(par,network):
    
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.n_in,par.nn)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=1/np.sqrt(par.n_in),
                                    a=par.init_a,b=par.init_b) 
    if par.init == 'uniform':
        network.w = nn.Parameter(torch.FloatTensor(par.n_in,par.nn).uniform_(0.,par.init_mean))
        
    if par.init == 'fixed':
        network.w = nn.Parameter(par.init_mean*torch.ones(par.n_in,par.nn))
        
    w_rec = par.w0_rec*np.ones((par.nn,par.nn))
    w_rec = np.where(np.eye(par.nn)>0,np.zeros_like(w_rec),w_rec)
    network.wrec = torch.as_tensor(w_rec,dtype=par.dtype).to(par.device)
    
    return network

def forward_nn_PyTorch(par,network,x):
    
    v,z = [], [[[] for n in range(par.nn)] for b in range(par.batch)]
    
    for t in range(par.T):            

        v.append(network.v)              

        'update of the neuronal variables - forward pass'
        network(x[:,t,:,:])   
        for b in range(par.batch):
            for n in range(par.nn):
                if network.z[b][n].item() != 0: z[b][n].append(t*par.dt)    
        
    return network, torch.stack(v,dim=1), z

def train_nn_PyTorch(par,network,x=None,timing=None):
    
    'set loss and optimizer'
    
    loss = nn.MSELoss(reduction='sum')
    
    'set optimizer for each neuron in the network'
    optimizerList = []
    for n in range(par.nn):
        if par.optimizer == 'Adam':
            optimizerList.append(torch.optim.Adam(network.parameters(),
                                  lr=par.eta,betas=(.9,.999)))
        elif par.optimizer == 'SGD':
            optimizerList.append(torch.optim.SGD(network.parameters(),lr=par.eta))
    
#    if par.optimizer == 'Adam':
#        optimizer = torch.optim.Adam(network.parameters(),
#                                  lr=par.eta,betas=(.9,.999))
#    elif par.optimizer == 'SGD':
#        optimizer = torch.optim.SGD(network.parameters(),lr=par.eta)

    'allocate outputs'
    loss_list = [[] for n in range(par.nn)]
    w_list = []
    v_list, spk_list = [], []        

    for e in range(par.epochs):
        
        if par.noise == True:

            'load or create data'            
            if par.upload_data == True:
                x = torch.load(par.load_dir+'x_data_{}.npy'.format(1))
            else: x = funs.get_multisequence_nn(par,timing)
    
        'initialize neuron state and solve dynamics (forward pass)'
        network.state()
        network, v, z = forward_nn_PyTorch(par,network,x)
        
        """
        optimization step: 
            - computes loss over the whole duration of the forward pass
            - optimize the model offline, if required
        """
            
        EList = []
        x_hatList = []
        for n in range(par.nn):
            x_hatList.append(torch.einsum("bt,j->btj",v[:,:,n],network.w[:,n]))
            EList.append(.5*loss(x_hatList[n],x[:,:,:,n]))

#        optimizer.zero_grad()
#        for n in range(par.nn):
#            EList[n].backward(retain_graph = True)
#        optimizer.step()
        
        for n in range(par.nn):
            optimizerList[n].zero_grad()
        for n in range(par.nn):
            EList[n].backward(retain_graph = True)
        for n in range(par.nn):
            optimizerList[n].step()
        
        'save output'
        v_list.append(v.detach().numpy())
        spk_list.append(z)
        w_list.append(network.w.detach().clone().numpy())
       
        if e%50 == 0: 
            print(network.w)
            print('epoch {} out of {}'.format(e,par.epochs))
    
    return w_list, v_list, spk_list, loss_list
'---------------------------------------------------------------------------'