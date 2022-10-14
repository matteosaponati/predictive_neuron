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
'numerical solution and training for sequences - NumPy version'

def initialize_weights_NumPy(par,neuron):
    if par.init == 'fixed': 
        return par.init_mean*np.ones(par.N)
    elif par.init == 'random':
        return stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.N)), 
                              (par.init_b-par.init_mean)/(1/np.sqrt(par.N)), 
                              loc=par.init_mean, scale=1/np.sqrt(par.N)).rvs(par.N)
    else: return neuron.w

def forward_NumPy(par,neuron,x): 
    
    v,z = [], []
    loss = []
    
    for t in range(par.T):  
        
        'compute prediction error - Equation 2'
        v.append(neuron.v)
        loss.append(np.linalg.norm(x[:,t] - neuron.v*neuron.w))
        
        'update weights and membrane potential - Equation 1 and Equation 3'
        neuron(x[:,t])          
        if neuron.z != 0: z.append(t*par.dt)  
        
    return neuron, v, z, loss

def train_NumPy(par,neuron,x=None,timing=None):
    
    w = []
    v_list, spk_list = [],[]
    loss_list = []
    
    for e in range(par.epochs):     
        
        if par.noise == True:
            x = funs.get_sequence_NumPy(par,timing,
                                             onset=par.onset_list[e])
            
        neuron.state()
        neuron, v, spk, loss = forward_NumPy(par,neuron,x)    
        
        v_list.append(v)
        spk_list.append(spk)
        loss_list.append(np.sum(loss))
        w.append(neuron.w)
        
        if e%100 == 0: print("""epoch {} out of {}""".format(e,par.epochs))      
        
    return w, v_list, spk_list, loss_list


'---------------------------------------------------------------------------'
'numerical solution and training for sequences - PyTorch version'

def initialize_weights_PyTorch(par,neuron):
    if par.init == 'random':
        neuron.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(neuron.w, mean=par.init_mean, std=.1/np.sqrt(par.N),
                                    a=par.init_a,b=par.init_b)
    if par.init == 'fixed':
        neuron.w = nn.Parameter(par.init_mean*torch.ones(par.N)).to(par.device)
    
    return neuron
    
def forward_PyTorch(par,neuron,x):
    
    v,z = [], []
    
    for t in range(par.T):            

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

        'update of the neuronal variables'
        neuron(x[:,t])        
        if neuron.z[0] != 0: z.append(t*par.dt)    
        
    return neuron, torch.stack(v,dim=1), z

def train_PyTorch(par,neuron,x=None,timing=None):
    
    'set loss and optimizer'
    
    loss = nn.MSELoss(reduction='sum')
    
    if par.optimizer == 'Adam':
        optimizer = torch.optim.Adam(neuron.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    elif par.optimizer == 'SGD':
        optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
    
    'allocate outputs'
    loss_list = []
    w_list = []
    v_list, spk_list = [], []        

    for e in range(par.epochs):
                
        'load input data'
        if par.noise == True:
            x = funs.get_sequence(par,timing)
    
            
        # x_data = np.load(par.loaddir+'sequence_Dt_{}_Nseq_{}_Ndist_{}_rep_{}.npy'.format(
        #                     par.Dt,par.N_seq,par.N_dist,np.random.randint(1000)))
        
        neuron.state()
        neuron, v, z = forward_PyTorch(par,neuron,x)
        
        """
        optimization step: 
            - computes loss over the whole duration of the forward pass
            - optimize the model offline, if required
        """
        
        x_hat = torch.einsum("bt,j->btj",v,neuron.w)
        E = .5*loss(x_hat,x)
        
        if par.optimizer != "online":
            optimizer.zero_grad()
            E.backward()
            optimizer.step()
        
        'save output'
        loss_list.append(E.item())
        w_list(neuron.w.detach().numpy())
        v_list.append(v.detach().numpy())
        spk_list.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return w_list, v_list, spk_list, loss_list


# '----------------'
# def forward(par,neuron,x_data):
    
#     v,z = [], []
    
#     for t in range(par.T):            

#         v.append(neuron.v)              

#         if par.optimizer == 'online':            
#             with torch.no_grad():
#                 neuron.backward_online(x_data[:,t])
#                 neuron.update_online()            

#         neuron(x_data[:,t])        
#         if neuron.z[0] != 0: z.append(t*par.dt)    
        
#     return neuron, torch.stack(v,dim=1), z
# '----------------'

# '-------------------'
# def train(par):
    
#     'fix seed'
#     torch.manual_seed(par.seed)
#     torch.cuda.manual_seed(par.seed)
#     np.random.seed(par.seed)
        
#     'set model'
#     neuron = models.NeuronClass(par)
#     loss = nn.MSELoss(reduction='sum')
    
#     'initialization'
#     if par.init == 'trunc_gauss':
#         neuron.w = nn.Parameter(torch.empty(par.N)).to(par.device)
#         torch.nn.init.trunc_normal_(neuron.w, mean=par.init_mean, std=.1/np.sqrt(par.N),
#                                     a=par.init_a,b=par.init_b)
#     if par.init == 'fixed':
#         neuron.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
    
#     'optimizer'
#     if par.optimizer == 'Adam':
#         optimizer = torch.optim.Adam(neuron.parameters(),
#                                   lr=par.eta,betas=(.9,.999))
#     elif par.optimizer == 'SGD':
#         optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
    
#     'allocate outputs'
#     loss_out = []
#     w = np.zeros((par.epochs,par.N))
#     v_out, spk_out = [], []

#     for e in range(par.epochs):
                
#         'load input data'
#         x_data = np.load(par.loaddir+'sequence_Dt_{}_Nseq_{}_Ndist_{}_rep_{}.npy'.format(
#                             par.Dt,par.N_seq,par.N_dist,np.random.randint(1000)))
        
#         neuron.state()
#         neuron, v, z = forward(par,neuron,x_data)
        
#         'evaluate loss'
#         x_hat = torch.einsum("bt,j->btj",v,neuron.w)
#         E = .5*loss(x_hat,x_data)
#         if par.optimizer != "online":
#             optimizer.zero_grad()
#             E.backward()
#             optimizer.step()
        
#         'output'
#         loss_out.append(E.item())
#         w[e,:] = neuron.w.detach().numpy()
#         v_out.append(v.detach().numpy())
#         spk_out.append(z)
        
#         if e%50 == 0: 
#             print('epoch {} loss {}'.format(e,E.item()))
    
#     return loss_out, w, v_out, spk_out
# '----------------'
# def forward_nn_selforg(par,network,x_data):
#     z = [[[] for b in range(par.batch)] for n in range(par.nn)]
#     v = []
#     for t in range(par.T):     
#         'append voltage state'
#         v.append(network.v.clone().detach().numpy())
#         'update weights online'
#         if par.online == True: 
#             with torch.no_grad():
#                 network.backward_online(x_data[:,t])
#                 network.update_online()  
#         'forward pass'
#         network(x_data[:,t]) 
#         'append output spikes'
#         for n in range(par.nn):
#             for b in range(par.batch):
#                 if network.z[b,n] != 0: z[n][b].append(t*par.dt)          
        
#     return network, np.stack(v,axis=1), z
# '----------------'

# def train_nn_selforg(par):
    
#     'create input data'
#     x_data = funs.get_sequence_nn_selforg(par)
    
#     'set model'
#     network = models.NetworkClass_SelfOrg(par)
    
#     'initialization'
#     if par.init == 'trunc_gauss':
#         network.w = nn.Parameter(torch.empty(par.n_in+par.lateral,par.nn)).to(par.device)
#         torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=.1/np.sqrt(par.par.n_in+par.lateral),
#                                     a=par.init_a,b=par.init_b) 
#         network.w[par.n_in:,] = par.w_0rec    
#     if par.init == 'fixed':
#         w = par.w_0*torch.ones(par.n_in+par.lateral,par.nn)
#         w[par.n_in:,] = par.w_0rec
#         network.w = nn.Parameter(w).to(par.device)
    
#     'allocate outputs'
#     w = np.zeros((par.epochs,par.n_in+par.lateral,par.nn))
#     z_out = [[] for n in range(par.nn)]
#     v_out = []
    
#     'training'
#     for e in range(par.epochs):
#         if e%50 == 0: print(e)  
            
#         network.state()
#         network, v, z = forward(par,network,x_data)
#         v_out.append(v)
        
#         w[e,:,:] = network.w.detach().numpy()
#         for n in range(par.nn):
#             z_out[n].append(z[n])

#     return w, z_out, v_out