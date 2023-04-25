import json
import sys
import os
import numpy as np
import torch

from utils.funs import get_Namespace_hyperparameters

'-----------------------------------------------------------------------------'

def main(path):
    
    f = open(path+'/hyperparameters.json')
    args = json.load(f)       
    par = get_Namespace_hyperparameters(args)
    
    'create log files'
    log = os.path.join(path, 'train.txt')
    with open(log,'w') as train:
        train.write('epoch, loss_train, loss_test \n')

    '-----------------------------'

    if par.package == 'NumPy':

        from models.NeuronClass import NeuronClassNumPy
        from utils.TrainerClassNumPy import TrainerClass

        loaddir = ('../_datasets/N_seq_{}_N_dist_{}_Dt_{}/'+
               'freq_{}_jitter_{}_onset_{}/').format(par.N_seq,par.N_dist,par.Dt,
                                             par.freq,par.jitter,par.onset)
        
        train_data = np.load(loaddir+'x_train.npy')
        test_data = np.load(loaddir+'x_test.npy')
        train_onset = np.load(loaddir+'onsets_train.npy')
        test_onset = np.load(loaddir+'onsets_test.npy')
        
        par.train_nb = int(train_data.shape[0]/par.batch)
        par.test_nb = int(test_data.shape[0]/par.batch)
        
        neuron = NeuronClassNumPy(par)
        neuron.initialize()

        'train'
        if par.onset > 1:
            trainer = TrainerClass(par,neuron,train_data,test_data,
                               train_onset,test_onset)
        else:
            trainer = TrainerClass(par,neuron,train_data,test_data)
        trainer.train(log)

    if par.package == 'PyTorch':
        
        from models.NeuronClass import NeuronClassPyTorch
        from utils.TrainerClassPyTorch import TrainerClass

        loaddir = ('../_datasets/N_seq_{}_N_dist_{}_Dt_{}/'+
               'freq_{}_jitter_{}_onset_{}/').format(par.N_seq,par.N_dist,par.Dt,
                                             par.freq,par.jitter,par.onset)
        
        train_data = np.load(loaddir+'x_train.npy')
        test_data = np.load(loaddir+'x_test.npy')
        train_data = torch.from_numpy(train_data).to(torch.float)
        test_data = torch.from_numpy(test_data).to(torch.float)

        if par.onset > 1 :
            train_onset = np.load(loaddir+'onsets_train.npy')
            test_onset = np.load(loaddir+'onsets_test.npy')
            train_onset = torch.from_numpy(train_onset).to(torch.float)
            test_onset = torch.from_numpy(test_onset).to(torch.float)
        
        par.train_nb = int(train_data.shape[0]/par.batch)
        par.test_nb = int(test_data.shape[0]/par.batch)

        neuron = NeuronClassPyTorch(par)
        neuron.initialize()

        if par.optimizer == 'SGD':
            optimizer = torch.optim.SGD(neuron.parameters(),lr=par.eta)
        if par.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(neuron.parameters(),lr=par.eta)
        if par.optimizer == 'Adam':
            optimizer = torch.optim.Adam(neuron.parameters(),lr = 1e-3,
                                         betas=(.9,.999))
        
        'train'
        if par.onset > 1:
            trainer = TrainerClass(par,neuron,optimizer,train_data,test_data,
                                   train_onset,test_onset)
        else:
            trainer = TrainerClass(par,neuron,optimizer,train_data,test_data)
        trainer.train(log)
    
    '-----------------------------'

    'save'    
    np.save(path+'loss_train',trainer.losstrainList)
    np.save(path+'loss_test',trainer.losstestList)
    np.save(path+'v',trainer.vList)
    np.save(path+'fr',trainer.frList)
    np.save(path+'z',trainer.zList)
    np.save(path+'onset',trainer.onsetList)
    np.save(path+'w',trainer.w)

'-----------------------------------------------------------------------------'

if __name__ == '__main__':   
    main(sys.argv[1])