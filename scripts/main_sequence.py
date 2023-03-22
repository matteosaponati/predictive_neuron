import json
import sys
import os
import numpy as np
import torch

from utils.funs import get_Namespace_hyperparameters
#from utils.data import SequencesDataset, prepare_dataloader

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
        
        par.train_nb = int(train_data.shape[0]/par.batch)
        par.test_nb = int(test_data.shape[0]/par.batch)
        
        neuron = NeuronClassNumPy(par)
        neuron.initialize()

        'train'
        trainer = TrainerClass(par,neuron,train_data,test_data)
        trainer.train(log)

    if par.package == 'PyTorch':
        
        from models.NeuronClass import NeuronClassPyTorch
        from utils.TrainerClassPyTorch import TrainerClass

        train_dataloader = SequencesDataset(par,mode='train')
        train_data, par.train_nb = prepare_dataloader(train_dataloader,
                                                      batch_size=par.batch)
        test_dataloader = SequencesDataset(par,mode='test')
        test_data, par.test_nb = prepare_dataloader(test_dataloader,
                                       batch_size=par.batch)
        
        neuron = NeuronClassPyTorch()
        parameters = [neuron.w]
        
        # define optimizer

        if par.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters,lr=par.eta_out)
        if par.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(parameters,lr=par.eta_out)
        if par.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters,lr = par.eta_out,
                                         betas=(.9,.999))
        
        'train'
        trainer = TrainerClass(par,neuron,train_data,test_data)
        trainer.train(log)
    
    '-----------------------------'

    'save'    
    np.save(path+'loss_train',trainer.losstrainList)
    np.save(path+'loss_test',trainer.losstestList)
    np.save(path+'v',trainer.vList)
    np.save(path+'fr',trainer.frList)
    np.save(path+'z',trainer.zList)
    np.save(path+'w',trainer.w)

'-----------------------------------------------------------------------------'

if __name__ == '__main__':   
    main(sys.argv[1])