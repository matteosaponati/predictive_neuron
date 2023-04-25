import argparse
import os
import numpy as np

from utils.funs import get_dir_results, get_hyperparameters
from models.NeuronClass import NeuronClassNumPy
from utils.TrainerClassNumPy import TrainerClass

'-------------------------------'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    par = parser.parse_args()

    par.name = 'multisequence'
    par.package = 'NumPy'

    par.bound = 'none'
    par.eta = 3e-7
    par.batch = 1
    par.epochs = 4000
    
    par.init = 'fixed'
    par.init_mean = .04
    
    par.sequence = 'deterministic'
    par.Dt = 2
    par.N_seq = 8
    par.n_afferents = 3
    par.delay = 20
    par.N = par.N_seq*par.n_afferents 

    par.freq = 5.
    par.jitter = 1.
    par.onset = 0

    par.dt = .05
    par.tau_m = 18.
    par.v_th = 2.6
    par.tau_x = 2.

    par.T = int((par.Dt*par.N_seq + par.n_afferents*par.delay + 
                     par.jitter)/(par.dt))
        
    par.dir_output = '../_results/'

    '-----------------------------------------'

    path = get_dir_results(par)
    if not os.path.exists(path): os.makedirs(path)
    get_hyperparameters(par,path)

    'create log files'
    log = os.path.join(path, 'train.txt')
    with open(log,'w') as train:
        train.write('epoch, loss_train, loss_test \n')
    
    loaddir = ('../_datasets/{}/N_seq_{}_n_afferents_{}_Dt_{}_delay_{}/'+
               'freq_{}_jitter_{}/').format(par.name,par.N_seq,par.n_afferents,
                                                     par.Dt,par.delay,par.freq,par.jitter)
        
    train_data = np.load(loaddir+'x_train.npy')
    test_data = np.load(loaddir+'x_test.npy')
        
    par.train_nb = par.batch
    par.test_nb = par.batch
        
    neuron = NeuronClassNumPy(par)
    neuron.initialize()

    'train'
    trainer = TrainerClass(par,neuron,train_data,test_data)
    trainer.train(log)

    np.save(path+'loss_train',trainer.losstrainList)
    np.save(path+'loss_test',trainer.losstestList)
    np.save(path+'v',trainer.vList)
    np.save(path+'fr',trainer.frList)
    np.save(path+'z',trainer.zList)
    np.save(path+'onset',trainer.onsetList)
    np.save(path+'w',trainer.w)