"main NN test"

import numpy as np
import torch
import torch.nn as nn
import os

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
    
    'get data'
    x_train, y_train, x_test, y_test = funs.get_fashionMNIST(par)
    
    'model'
    network = models_nn.StandardNetworkClass(par)
    readout = models_nn.ReadoutClass(par)
    
    ## REMEMBER HERE TO ADD CASE WHEN NETWORK IS RECURRENT
    'optimizer'
    if par.optimizer == 'Adam':
        optimizer = torch.optim.Adam([network.w,readout.w],
                                  lr=par.eta,betas=(.9,.999))
    elif par.optimizer == 'SGD':
        optimizer = torch.optim.SGD([network.w,readout.w],lr=par.eta)
    
    'allocate outputs'
    loss_out, accuracy_out = [], []

    for e in range(par.epochs):
        
        loss_batch, accuracy_batch = [],[]
                    
        for x_local,y_local in funs.sparse_from_fashionMNIST(par,x_train,y_train):
            
            y_hat, z = do_epoch(x_local.to_dense(),par,network,readout)
            
            'loss_evaluation'
            loss = loss_classification(y_hat,y_local)
            loss_batch.append(loss.item())
    
            'optimization'
            optimizer.zero_grad()        
            loss.backward()
            optimizer.step()  
            
            with torch.no_grad():
                accuracy = do_accuracy(par,x_local.to_dense(),y_local,
                                                  network,readout)
                accuracy_batch.append(accuracy)
        
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
    z, y_hat = [], []
    
    '------------------------'
    for t in range(par.T):    

        network(x_data[:,t]) 
        readout(network.z)
        
        y_hat.append(readout.y)
        z.append(network.z)
    '------------------------'

    z = torch.stack(z,dim=1)
    y_hat = torch.stack(y_hat,dim=1)
    
    return y_hat, z.detach()

def do_accuracy(par,x_data,y_data,network,readout):
    
    network.state()
    readout.state()
    y_hat = []
    '------------------------'
    for t in range(par.T):    

        network(x_data[:,t]) 
        readout(network.z)
        
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
                        shallow hidden layer on classification task, SHD dataset
                        """
                        )
    
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['SGD','NAG','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--eta',type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--N', type=int, default=256,
                        help='number of batches')
    
    parser.add_argument('--reg_fr', type=int, default=10,
                        help='target pop firing rate [Hz]')
    parser.add_argument('--reg_coeff', type=int, default=1)
    
    parser.add_argument('--rep',type=int,default=1,
                        help='iteration number')
    parser.add_argument('--save_output',type=str,default='False',
                        help='set output save')
    
    'architecture'
    parser.add_argument('--n_in', type=int, default=28*28) 
    parser.add_argument('--n', type=int, default=100) 
    parser.add_argument('--n_out', type=int, default=10)
    parser.add_argument('--data_augment',type=str,default='False',
                        help='set data augmentation')
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
    par.alpha = float(np.exp(-par.dt/par.tau_m))
    par.prompt = 1
    
    'train'
    loss_class, accuracy = train(par)
    
    '-----------'
    'save'
    par.dtype = 'torch.float'
    par.savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/nn/'
    
    np.save(os.path.join(par.savedir,'accuracy'),accuracy)
    np.save(os.path.join(par.savedir,'loss_class'),loss_class)
