import argparse
import os
import numpy as np

from utils.funs import get_dir_results, get_hyperparameters
from models.SelfOrgNetworkClass import NetworkClassNumPy
from utils.TrainerClassNumPy_SelfOrg import TrainerClass

'-------------------------------'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    par = parser.parse_args()

    par.name = 'selforg'
    par.network_type = 'nearest'
    par.package = 'NumPy'

    par.bound = 'none'
    par.eta = 1e-6
    par.batch = 1
    par.epochs = 2000
    
    par.init = 'fixed'
    par.init_mean = .02
    par.init_rec = .0003
    
    par.Dt = 2
    par.n_in = 8
    par.nn = 10
    par.delay = 4

    par.freq = 5.
    par.jitter = 1.

    par.dt = .05
    par.tau_m = 25.
    par.v_th = 3.1
    par.tau_x = 2.

    par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                     par.jitter + 80)/(par.dt))
    
    par.N = par.n_in+2
    
    par.dir_output = '../_results/'

    '-----------------------------------------'

    path = get_dir_results(par)
    if not os.path.exists(path): os.makedirs(path)
    get_hyperparameters(par,path)

    'create log files'
    log = os.path.join(path, 'train.txt')
    with open(log,'w') as train:
        train.write('epoch, loss_train, loss_test \n')
    
    loaddir = ('../_datasets/{}/{}/n_in_{}_nn_{}_Dt_{}/').format(par.name,par.network_type,par.n_in,par.nn,par.Dt) + \
                    ('freq_{}_jitter_{}/').format(par.freq,par.jitter)
        
    train_data = np.load(loaddir+'x_train.npy')
    test_data = np.load(loaddir+'x_test.npy')
        
    ## complete online training: one example per batch
    par.train_nb = par.batch
    par.test_nb = par.batch
        
    network = NetworkClassNumPy(par)
    network.initialize()
    print(network.w)

    'train'
    trainer = TrainerClass(par,network,train_data,test_data)
    trainer.train(log)

    np.save(path+'loss_train',trainer.losstrainList)
    np.save(path+'loss_test',trainer.losstestList)
    np.save(path+'v',trainer.vList)
    np.save(path+'activity',trainer.activityList)
    np.save(path+'z',trainer.zList)
    np.save(path+'w',trainer.w)