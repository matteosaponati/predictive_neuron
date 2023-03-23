import json
import sys
import os
import numpy as np
import torch

from utils.funs import get_Namespace_hyperparameters
from utils.data import SequencesDataset, prepare_dataloader

'-----------------------------------------------------------------------------'

def main(path):
    
    f = open(path+'hyperparameters.json')
    args = json.load(f)       
    par = get_Namespace_hyperparameters(args)
    
    'create log files'
    log = os.path.join(path, 'train.txt')
    with open(log,'w') as train:
        train.write('epoch, loss_train, loss_test \n')

    '-----------------------------'

    if par.package == 'NumPy':

        if par.network_type == 'random':

            from models.RandomNetworkClass import NetworkClassNumPy 
            from utils.TrainerClassNumPy_SelfOrg import TrainerClass

            loaddir = ('../_datasets/{}/{}/n_in_{}_nn_{}_Dt_{}/').format(par.name,par.network_type,
                                                                         par.n_in,par.nn,par.Dt) + \
                    'freq_{}_jitter_{}/'.format(par.freq,par.jitter)
        
            train_data = np.load(loaddir+'x_train_{}.npy'.format(par.type))
            test_data = np.load(loaddir+'x_test_{}.npy'.format(par.type))
        
            ## complete online training: one example per batch
            par.train_nb = par.batch
            par.test_nb = par.batch
            
            network = NetworkClassNumPy()
            network.get_mask()
            network.initialize()
            
            'train'
            trainer = TrainerClass(par,network,train_data,test_data)
            trainer.train(log)

    if par.package == 'PyTorch':
        
        from models.SelfOrgNetworkClass import NetworkClassPyTorch
        from utils.TrainerClassPyTorch_SelfOrg import TrainerClass

        train_dataloader = SequencesDataset(par,mode='train')
        train_data, par.train_nb = prepare_dataloader(train_dataloader,
                                                      batch_size=par.batch)
        test_dataloader = SequencesDataset(par,mode='test')
        test_data, par.test_nb = prepare_dataloader(test_dataloader,
                                       batch_size=par.batch)
        
        network = NetworkClassPyTorch()
        parameters = [network.w]

        if par.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters,lr=par.eta_out)
        if par.optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(parameters,lr=par.eta_out)
        if par.optimizer == 'Adam':
            optimizer = torch.optim.Adam(parameters,lr = par.eta_out,
                                         betas=(.9,.999))
        
        'train'
        trainer = TrainerClass(par,network,train_data,test_data)
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