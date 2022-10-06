"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn.py"
neural network trained on classification task

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from predictive_neuron import models_nn, funs

'-------------------'

def loss_classification(y_hat,y_data):
    loss = nn.NLLLoss()
    m,_ = torch.max(y_hat,1)
    log_y_hat = nn.functional.log_softmax(m,dim=1)
    return loss(log_y_hat,y_data)

'-------------------'
def train(par):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'alpha filter'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to("cpu") 
    
    'get data'
    x_train, y_train, x_test, y_test = funs.get_fashionMNIST(par)
    
    'model'
    network = models_nn.NetworkClass(par)
    readout = models_nn.ReadoutNNClass(par)
    
    loss_fn = nn.MSELoss(reduction='sum')

    '----------'
    'optimizers NN'
    optimizerList = []
    for n in range(par.nn):
        optimizer = torch.optim.Adam(network.parameters(),
                                  lr=par.eta,betas=(.9,.999))
        optimizerList.append(optimizer)
    'optimizer readout'
    if par.optimizer == 'Adam':
        optimizer = torch.optim.Adam(readout.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    elif par.optimizer == 'SGD':
        optimizer = torch.optim.SGD(readout.parameters(),lr=par.eta)
    '----------'
    
    'allocate outputs'
    loss_out, accuracy_out = [], []
    loss_nn = [[] for n in range(par.nn)]
    
    for e in range(par.epochs):
        
        loss_batch, accuracy_batch = [],[]
                    
        for x_local,y_local in funs.sparse_from_fashionMNIST(par,x_train,y_train):
            
            x_data = F.conv1d(x_local.to_dense().permute(0,2,1),filter.expand(par.n_in,-1,-1),
                     padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
            
            print('convolution done')
            
            y_hat, v, z = do_epoch(x_data.permute(0,2,1),par,network,readout)
            
            '----------'
            'optimization NN'
            x_hat = torch.einsum("btn,jn->btjn",v,network.w)
            lossList = []
            for n in range(par.nn):  
                loss = loss_fn(x_hat[:,:,:,n],x_data.permute(0,2,1))
                lossList.append(loss)
            for n in range(par.nn): 
                lossList[n].backward(retain_graph = True)
                loss_nn[n].append(lossList[n].item())
            for n in range(par.nn): 
                 optimizerList[n].step()
            'optimization readout '
            loss = loss_classification(y_hat,y_local)
            loss_batch.append(loss.item())
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()  
            '----------'
            
            '----------'
            'accuracy'
            with torch.no_grad():
                accuracy = do_accuracy(par,x_local.to_dense(),y_local,
                                                  network,readout)
                accuracy_batch.append(accuracy)
            '----------'
                
        'save output'
        loss_out.append(np.mean(loss_batch))
        accuracy_out.append(np.mean(accuracy_batch))
        
        'prompt output'
        if e%par.prompt == 0:
            print("""
                  epoch {} ; 
                  loss class: {:.2g}
                  accuracy: {:.2g}
                  """.format(e,loss_out[e],accuracy_out[e]))

    return loss_out, accuracy_out

def do_epoch(x_data,par,network,readout):
    
    network.state()
    readout.state()
    v,z = [], []
    y_hat = []
    
    '------------------------'
    for t in range(par.T):    

        v.append(network.v)   
        
        network(x_data[:,t]) 
        readout(network.z.detach())
        
        y_hat.append(readout.y)
        z.append(network.z_out.detach())
    '------------------------'
    
    return torch.stack(y_hat,dim=1), torch.stack(v,dim=1), torch.stack(z,dim=1)

def do_accuracy(par,x_data,y_data,network,readout):
    
    network.state()
    readout.state()
    y_hat = []
    '------------------------'
    for t in range(par.T):    

        network(x_data[:,t]) 
        readout(network.z.detach())
        
        y_hat.append(readout.y)
    
    y_hat = torch.stack(y_hat,dim=1)
    m,_ = torch.max(y_hat,1)
    _,am = torch.max(m,1)

    acc = np.mean((y_data==am).cpu().detach().cpu().numpy())
    acc = np.mean((y_data==am).detach().cpu().numpy())
        
    return acc

'-------------------'

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                        description="""
                        NN classification task, fashionMNIST
                        """
                        )
    
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['SGD','NAG','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--eta',type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--is_rec',type=str,default='False')
    parser.add_argument('--N', type=int, default=256,
                        help='number of batches')
    
    'architecture'
    parser.add_argument('--n_in', type=int, default=28*28) 
    parser.add_argument('--nn', type=int, default=100) 
    parser.add_argument('--n_out', type=int, default=10)
    parser.add_argument('--T', type=int, default=100) 
    
    'neuronal model'
    parser.add_argument('--dt', type=float, default= 1.) 
    parser.add_argument('--tau_m', type=float, default= 20.) 
    parser.add_argument('--v_th', type=float, default= 1.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    
    'additional parameters'
    par.device = "cuda" if torch.cuda.is_available() else "cpu"
    par.device = "cpu"
    par.tau_x = 2.
    par.prompt = 1
    
    'train'
    loss_class, accuracy = train(par)
    
    '-----------'
    'save'
    par.dtype = 'torch.float'
    par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/nn/'
    
    np.save(os.path.join(par.savedir,'accuracy'),accuracy)
    np.save(os.path.join(par.savedir,'loss_class'),loss_class)