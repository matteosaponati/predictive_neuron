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
import torch
import torch.nn as nn

from predictive_neuron import funs

'---------------------------------------------------------------------------'
'numerical solution and training for sequences - PyTorch version'

'--------'

def initialize_weights_nn_PyTorch(par,network):
    
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.n_in,par.nn)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=1/np.sqrt(par.n_in),
                                    a=par.init_a,b=par.init_b) 
    if par.init == 'uniform':
        network.w = nn.Parameter(torch.FloatTensor(par.n_in,par.nn).uniform_(0.,par.init_mean))
        
    if par.init == 'fixed':
        network.w = par.init_mean*torch.ones(par.n_in,par.nn)
        
    w_rec = par.w0_rec*np.ones((par.nn,par.nn))
    w_rec = np.where(np.eye(par.nn)>0,np.zeros_like(w_rec),w_rec)
    network.wrec = nn.Parameter(torch.as_tensor(w_rec,dtype=par.dtype).to(par.device))
    
    return network

def forward_nn_PyTorch(par,network,x):
    
    v,z = [], [[[] for n in range(par.nn)] for b in range(par.batch)]
    
    for t in range(par.T):            

        v.append(network.v.detach().clone())              

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

        # check this         
        EList = []
        x_hatList = []
        for n in range(par.nn):
            x_hatList.append(torch.einsum("bt,j->btj",v[:,:,n],network.w[:,n]))
            EList.append(.5*loss(x_hatList[n],x[:,:,:,n]))
        
        if par.optimizer != "online":
            for n in range(par.nn):
                optimizerList[n].zero_grad()
            for n in range(par.nn):
                EList[n].backward(retain_graph = True)
            for n in range(par.nn):
                optimizerList[n].step()
        
        'save output'
        v_list.append(v.numpy())
        spk_list.append(z)
        w_list.append(network.w.detach().clone().numpy())
#        
        if e%50 == 0: 
            print(network.w)
            print('epoch {} out of {}'.format(e,par.epochs))
    
    return w_list, v_list, spk_list, loss_list
'---------------------------------------------------------------------------'