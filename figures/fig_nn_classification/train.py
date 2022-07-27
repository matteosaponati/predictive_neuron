"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-

----------------------------------------------
"train.py"

training and test process
----------------------------------------------
Author:
s    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
import torch
from time import time
from src import models 
import os

import funs

def train(par):
    
    'set saving directory'
    savedir = os.path.join(par.path,'T_{}_nin_{}_n_{}_tau_{}'\
                                       .format(par.T,par.n_in,par.n,par.tau_m),
                                       'output')
    if not os.path.exists(savedir): os.makedirs(savedir)
    
    'get data'
    x_train, y_train, x_test, y_test = funs.get_SHD_dataset()
    
    'model'
    hidden = models.LIF_cell(par,is_rec=True)
    readout = models.readout_cell(par)
    
    if par.optimizer == 'SGD':
        optimizer = torch.optim.SGD([hidden.w,hidden.wrec,readout.w],
                                     lr = par.l_rate) 
    elif par.optimizer == 'NAG':
        optimizer = torch.optim.SGD([hidden.w,hidden.wrec,readout.w],
                                     lr = par.l_rate,
                                     momentum=.9,nesterov=True) 
    elif par.optimizer == 'Adam':
        optimizer = torch.optim.Adam([hidden.w,hidden.wrec,readout.w],
                                     lr = par.l_rate,
                                     betas = (.9,.999))
    else:
        raise NameError("ERROR: optimizer "+str(par.optimizer)+"not supported")
        
    E_class_out, E_reg_out, accuracy = [], [], []
    runtime = []
    
    'training'
    for e in range(par.epochs):
        
        E_class_local,E_reg_local, accuracy_local = [],[],[]
        labels = []        
        spk_times = [[] for k in range(par.n)]
        
        '----------------------------------'
        'get network parameters'
#        if par.save_output == 'True':
        np.save(os.path.join(savedir,'wrec_epoch_{}'.format(e)),hidden.wrec.cpu().detach().numpy())
        np.save(os.path.join(savedir,'win_epoch_{}'.format(e)),hidden.w.cpu().detach().numpy())
        np.save(os.path.join(savedir,'wout_epoch_{}'.format(e)),readout.w.cpu().detach().numpy())
        '----------------------------------'

        '----------------------------------'        
        for x_local,y_local in funs.sparse_data_from_hdf5(par,x_train,y_train):
                        
            z = do_epoch(x_local.to_dense(),y_local,par,hidden,readout,
                     optimizer,accuracy,E_class_local,E_reg_local,
                     runtime)
            with torch.no_grad():
                accuracy_local.append(do_accuracy(par,x_local.to_dense(),y_local,
                                                  hidden,readout))
        
            '--------'        
            if par.save_output == 'True':
                with torch.no_grad():
                    labels.extend(y_local.cpu().detach().numpy().flatten().tolist())
                    for k in range(par.n):
                        for b in range(par.N):
                            spk_times[k].append(torch.nonzero(z[b,:,k]).cpu().detach()\
                                         .numpy().flatten().tolist())              
            '--------'
        
        '----------------------------------'
        'get spike patterns and output variables' 
        if par.save_output == 'True':
            np.save(os.path.join(savedir,'spk_times_epoch_{}'.format(e)),
                    spk_times) 
            np.save(os.path.join(savedir,'labels_epoch_{}'.format(e)),
                    labels)
            del spk_times, labels
        
        E_class_out.append(np.mean(E_class_local))
        E_reg_out.append(np.mean(E_reg_local))
        accuracy.append(np.mean(accuracy_local))
        '----------------------------------'

        'prompt output'
        if e%par.prompt == 0:
            print("""
                  epoch {} ; E_class: {:.2g} \t E_reg: {:.2g}
                  comp. time (s) training {:.2g}
                  """.format(e,E_class_out[e],E_reg_out[e],runtime[e]))

    'test accuracy'
    with torch.no_grad():
        accuracy_test = []
        for x_local,y_local in funs.sparse_data_from_hdf5(par,x_test,y_test):
            accuracy_test.append(do_accuracy(par,x_local.to_dense(),y_local,
                                        hidden,readout))
        accuracy_test = np.mean(accuracy_test)
    
    print('training accuracy: {}'.format(accuracy[-1]))
    print('test accuracy: {}'.format(accuracy_test))
       
    return accuracy, runtime, E_class_out, E_reg_out, accuracy_test

'----------'

def do_epoch(x_data,y_data,par,hidden,readout,optimizer,
             accuracy,E_class_local,E_reg_local,runtime):
    
    state = hidden.state(par)
    out = readout.state(par)

    z, y_hat = [], []
    
    '------------------------'
    t_ref = time()
    for t in range(par.T):    

        state = hidden(x_data[:,t],state)
        out = readout(state.z,out)     
        y_hat.append(out.y)
        z.append(state.z)
    '----------------------------------------'

    z = torch.stack(z,dim=1)
    y_hat = torch.stack(y_hat,dim=1)
        
    'loss_evaluation'
    E_class = funs.loss_classification(y_hat,y_data)
#    E_reg = funs.reg_fr_population(par,z)
#    E_tot = E_class
    
    E_class_local.append(E_class.item())
#    E_reg_local.append(E_reg.item())
    
    'optimization'
    optimizer.zero_grad()        
#    E_tot.backward()
    E_class.backward()
    runtime.append(time()-t_ref)
    optimizer.step()  
    
    return z.detach()
    
def do_accuracy(par,x_data,y_data,hidden,readout):
    
    state = hidden.state(par)
    out = readout.state(par)
    y_hat = []
    '------------------------'
    for t in range(par.T):    

        state = hidden(x_data[:,t],state)
        out = readout(state.z,out)     
        y_hat.append(out.y)
    
    y_hat = torch.stack(y_hat,dim=1)
    m,_ = torch.max(y_hat,1)
    _,am = torch.max(m,1)

    acc = np.mean((y_data==am).cpu().detach().cpu().numpy())
    acc = np.mean((y_data==am).detach().cpu().numpy())
        
    return acc
