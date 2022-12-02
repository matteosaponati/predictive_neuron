"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"funs_train.py"
auxiliary functions to train the single neuron and the network model

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

from predictive_neuron import funs

'---------------------------------------------------------------------------'
'---------------------------------------------------------------------------'
'numerical solution and training for sequences - single neurons'

def initialize_weights_NumPy(par,neuron):
    """
    initialize the synaptic weights of the neuron model
    """
    
    if par.init == 'fixed': 
        return par.init_mean*np.ones(par.N)
    elif par.init == 'random':
        return stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.N)), 
                              (par.init_b-par.init_mean)/(1/np.sqrt(par.N)), 
                              loc=par.init_mean, scale=1/np.sqrt(par.N)).rvs(par.N)
    else: return neuron.w

def forward_NumPy(par,neuron,x): 
    """
    set list of membrane voltage, output spike variables and loss
    update the state variables at each timestep t and save values in lists
    """
    
    v,z,loss = [], [], []
    'forward step across time'    
    for t in range(par.T):  
        'compute prediction error - Equation 2'
        v.append(neuron.v)
        loss.append(np.linalg.norm(x[:,t] - neuron.v*neuron.w))
        'update weights and membrane potential - Equation 1 and Equation 3'
        neuron(x[:,t])          
        if neuron.z != 0: z.append(t*par.dt)  
        
    return neuron, v, z, loss

def train_NumPy(par,neuron,x=None,timing=None):
    """
    set if data is uploaded or created, depending on the type of inputs
    
    optimization step:
    - compute forward pass
    - compute online backward pass
    
    save relevant variables
    """
    
    'allocate outputs'
    w = []
    v_list, spk_list, loss_list = [],[],[]
    
    'train network across epochs'
    for e in range(par.epochs):     
        
        'load or create data'
        if par.upload_data == 1:
             x = np.load(par.load_dir+'x_NumPy_{}.npy'.format(np.random.randint(1000)))
        else:
            if par.noise == 1:
                if par.name == 'sequence': 
                    x = funs.get_sequence_NumPy(par,timing,onset=par.onset_list[e])
                if par.name == 'multisequence': 
                    x = funs.get_multisequence_NumPy(par,timing)
                if par.name == 'rhythms':
                    x = funs.get_rhythms_NumPy(par,timing)
        
        'initialize neuron state and solve dynamics (forward+backward pass)'
        neuron.state()
        neuron, v, spk, loss = forward_NumPy(par,neuron,x)    
        
        'save output'
        v_list.append(v)
        spk_list.append(spk)
        loss_list.append(np.sum(loss))
        w.append(neuron.w)
        
        if e%100 == 0: print("""epoch {} out of {}""".format(e,par.epochs))      
        
    return w, v_list, spk_list, loss_list

'--------------------------------'

def initialize_weights_PyTorch(par,neuron):
    """
    initialize the synaptic weights of the neuron model
    """
    
    if par.init == 'random':
        neuron.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(neuron.w, mean=par.init_mean, std=1/np.sqrt(par.N),
                                    a=par.init_a,b=par.init_b)
    if par.init == 'fixed':
        neuron.w = nn.Parameter(par.init_mean*torch.ones(par.N)).to(par.device)
    
    return neuron
    
def forward_PyTorch(par,neuron,x):
    """
    set list of membrane voltage, output spike variables and loss
    update the state variables at each timestep t 
    optimize online at each time step or at the end of forward pass
    nd save values in lists
    """
    
    v,z = [], [[] for b in range(par.batch)]
    'forward step across time'  
    for t in range(par.T):            
        'get voltage at timestep t-1'
        v.append(neuron.v)              
        """
        online optimization step:
            - computes the estimate of the gradient at timestep t
            - updates the synaptic weights online
        """
        if par.optimizer == 'online':            
            with torch.no_grad():
                neuron.backward_online(x[:,t])
                neuron.update_online()            
        'update of the neuronal variables - forward pass'
        neuron(x[:,t])   
        for b in range(par.batch):
            if neuron.z[b] != 0: z[b].append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z

def train_PyTorch(par,neuron,x=None,timing=None):
    """
    define the MSE loss function
    
    optimization step :
    - set optimizer
    - compute forward pass
    - if not online, compute the MSE loss and run optimizer.step()
    
    save relevant variables
    """
    
    'define loss function and optimizers'
    loss = nn.MSELoss(reduction='sum')
    if par.optimizer == 'Adam':
        optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    elif par.optimizer == 'SGD':
        optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
    
    'allocate outputs'
    loss_list, w_list = [], []
    v_list, spk_list = [], []        

    'train network across epochs'
    for e in range(par.epochs):
        
        'load or create data'
        if par.upload_data == 1:
             x = np.load(par.load_dir+'x_NumPy_{}.npy'.format(np.random.randint(1000)))
        if par.noise == 1:
            if par.name == 'sequence': 
                x,_ = funs.get_sequence(par,timing)
            if par.name == 'multisequence': 
                x = funs.get_multisequence(par,timing)
            
        'initialize neuron state and solve dynamics (forward pass)'
        neuron.state()
        neuron, v, z = forward_PyTorch(par,neuron,x)
        
        'optimization step'
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x)
        if par.optimizer != "online":
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
        
        'save output'
        loss_list.append(E.item())
        w_list.append(neuron.w.detach().clone().numpy())
        v_list.append(v.detach().numpy())
        spk_list.append(z)
        
        if e%50 == 0: 
            print(neuron.w)
            print('epoch {} loss {}'.format(e,E.item()/par.T))
    
    return w_list, v_list, spk_list, loss_list

'---------------------------------------------------------------------------'
'---------------------------------------------------------------------------'

def initialization_weights_nn_NumPy(par,network):
    """
    initialize the synaptic weights of the network model
    - random: weights to external inputs follows a truncated gaussian dist, 
     recurrent weights are fixed 
     - fixed: all weiights are fixed
    """
    
    if par.init == 'random':
        network.w = stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.n_in+par.lateral)), 
                              (par.init_b-par.init_mean)/(1/np.sqrt(par.n_in+par.lateral)), 
                              loc=par.init_mean, scale=1/np.sqrt(par.n_in+par.lateral)).rvs((par.n_in+par.lateral,par.nn))
        network.w[par.n_in:,] = par.w_0rec    
        
    if par.init == 'fixed':
        network.w = par.init_mean*np.ones((par.n_in+par.lateral,par.nn))
        network.w[par.n_in:,] = par.w_0rec
    
    return network

def initialization_weights_nn_AlltoAll(par,network):
    """
    initialize the synaptic weights of the network model
    - random: weights to external inputs follows a truncated gaussian dist, 
     recurrent weights are fixed 
     - fixed: all weiights are fixed
    """
    
    if par.init == 'random':
        network.w = stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.n_in+par.nn)), 
                              (par.init_b-par.init_mean)/(1/np.sqrt(par.n_in+par.nn)), 
                              loc=par.init_mean, scale=1/np.sqrt(par.n_in+par.nn)).rvs((par.n_in+par.nn,par.nn))
        network.w[par.n_in:,] = par.w_0rec    
        
    if par.init == 'fixed':
        network.w = par.init_mean*np.ones((par.n_in+par.nn,par.nn))
        network.w[par.n_in:,] = par.w_0rec
    
    return network

def forward_nn_NumPy(par,network,x_data):
    """
    set list of membrane voltage and output spike variables for each neuron
    update the state variables at each timestep t and save values in lists
    """
    
    z = [[] for n in range(par.nn)]
    v = []
    'forward pass across time'
    for t in range(par.T):     
        'get voltage at timestep t-1'
        v.append(network.v)
        'update weights and membrane potential'
        network(x_data[:,t]) 
        'get output spike at timestep t'
        for n in range(par.nn):
            if network.z[n] != 0: z[n].append(t*par.dt)          
        
    return network, np.stack(v,axis=1), z

def train_nn_NumPy(par,network,x=None,timing=None):
    """
    set if data is uploaded or created, depending on the type of inputs
    
    optimization step:
    - compute forward pass
    - compute online backward pass
    
    save relevant variables
    """
        
    'allocate outputs'
    w = []
    v_list, z_list = [], [[] for n in range(par.nn)]
    
    'train network across epochs'
    for e in range(par.epochs):
        
        'load or create data'
        if par.noise == True:      
            if par.upload_data == True:
                x = np.load(par.load_dir+'x_data_{}.npy'.format(np.random.randint(1000)))
            else:
                x = funs.get_sequence_nn_selforg_NumPy(par,timing)
        
        'initialize network state and solve dynamics (forward+backward pass)'
        network.state()
        network, v, z = forward_nn_NumPy(par,network,x)
        
        'save output'
        v_list.append(v)
        w.append(network.w)
        for n in range(par.nn):
            z_list[n].append(z[n])
        
        if e%100 == 0: print("""epoch {} out of {}""".format(e,par.epochs))  

    return w, v_list, z_list

'--------------------------------'

def initialize_weights_nn_PyTorch(par,network):
    """
    initialize the synaptic weights of the network model
    - random: weights to external inputs follows a truncated gaussian dist, 
     recurrent weights are fixed 
     - fixed: all weiights are fixed
    """
    
    if par.init == 'random':
        network.w = nn.Parameter(torch.empty(par.n_in+par.lateral,par.nn)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=1/np.sqrt(par.n_in+par.lateral),
                                    a=par.init_a,b=par.init_b) 
        network.w[par.n_in:,] = par.w_0rec    
    if par.init == 'fixed':
        w = par.init_mean*torch.ones(par.n_in+par.lateral,par.nn)
        w[par.n_in:,] = par.w_0rec
        print(w)
        network.w = nn.Parameter(w).to(par.device)
    
    return network

def forward_nn_PyTorch(par,network,x_data):
    """
    set list of membrane voltage and output spike variables for each neuron
    update the state variables at each timestep t and save values in lists
    """
    
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    'forward pass across time'
    for t in range(par.T):     
        'get voltage at timestep t-1'
        v.append(network.v.clone().detach().numpy())
        """
        online optimization step - backward pass:
            - computes the estimate of the gradient at timestep t
            - updates the synaptic weights online
        """
        if par.online == True: 
            with torch.no_grad():
                network.backward_online(x_data[:,t,:,:])
                network.update_online()  
        'update of the neuronal variables - forward pass'
        network(x_data[:,t]) 
        'get output spike at timestep t'
        for n in range(par.nn):
            for b in range(par.batch):
                if network.z[b,n] != 0: z[n][b].append(t*par.dt)          
        
    return network, np.stack(v,axis=1), z

def train_nn_PyTorch(par,network,x=None,timing=None):
    """
    optimization step:
    - compute forward pass
    - compute online backward pass
    
    save relevant variables
    """
        
    'allocate outputs'
    w = np.zeros((par.epochs,par.n_in+par.lateral,par.nn))
    z_out = [[] for n in range(par.nn)]
    v_out = []
    
    'train network across epochs'
    for e in range(par.epochs):
        
        'create data'
        if par.noise == True:            
            x = funs.get_sequence_nn_selforg(par,timing)

        'initialize network state and solve dynamics (forward+backward pass)'
        network.state()
        network, v, z = forward_nn_PyTorch(par,network,x)
        v_out.append(v)
        
        'save output'
        w[e,:,:] = network.w.detach().numpy()
        for n in range(par.nn):
            z_out[n].append(z[n])
        if e%50 == 0: print(e)  

    return w, v_out, z_out