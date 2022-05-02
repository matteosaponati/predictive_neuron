import os
import h5py
import numpy as np
import torch
import torch.nn as nn

def sparse_data_from_hdf5(par,x_data,y_data,max_time=1.4,shuffle=True):
    """ 
    this generator takes a spike dataset and generates spiking network input as sparse tensors. 
    args:
        x_data: ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y_data: labels
    """
    
    'get labels and batch size'
    labels_ = np.array(y_data,dtype=np.int)
    number_of_batches = len(labels_)//par.N
    sample_index = np.arange(len(labels_))

    if shuffle:
        np.random.shuffle(sample_index)
    
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[par.N*counter:par.N*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            
            'data augmentation'
            if par.data_augment == 'True':
                times,units = data_augmentation(par,x_data,idx,max_time)
            else:
                times = x_data['times'][idx] 
                units = x_data['units'][idx]
            
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

'-------------------------'

def get_SHD_dataset():
    
    cache_dir = os.path.expanduser("~/snn-neuroscience-methods/data")
    train_file = h5py.File(os.path.join(cache_dir,'SHD_dataset','shd_train.h5'),'r')
    test_file = h5py.File(os.path.join(cache_dir,'SHD_dataset','shd_test.h5'),'r')
    
    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    
    return x_train, y_train, x_test, y_test

def data_augmentation(par,x_data,idx,max_time):
    
    firing_times = x_data['times']
    units_fired = x_data['units']

    'binning in time'
    time_bins = np.linspace(0, max_time, num=par.T)
    times = np.digitize(firing_times[idx], time_bins)
    'binning in space'
    space_bins = np.linspace(0, 700, par.n_in)
    units = np.digitize(units_fired[idx],space_bins)

    return times, units

'-------------------------'

def loss_classification(y_hat,y_data):
    loss = nn.NLLLoss()
    m,_ = torch.max(y_hat,1)
    log_y_hat = nn.functional.log_softmax(m,dim=1)
    return loss(log_y_hat,y_data)

def reg_fr_population(par,z):
    fr_avg = torch.mean(z,dim=(0,1))/par.dt
    return par.reg_coeff*torch.mean(torch.square(torch.max(fr_avg - par.reg_fr*(1e3/par.T))))

def reg_spikes_L1(coeff,z):
    return coeff*torch.sum(z)
def reg_spikes_L2(coeff,z):
    return coeff*torch.mean(torch.sum(torch.sum(z,dim=0),dim=0)**2)

'-------------------------'


