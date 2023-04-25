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

    par.name = 'sequence'
    par.package = 'NumPy'

    par.bound = 'soft'
    par.eta = 5e-4
    par.batch = 1
    par.epochs = 1000
    
    par.init = 'fixed'
    par.init_mean = .1
    
    par.sequence = 'deterministic'
    par.Dt = 2
    par.N_seq = 100
    par.N_dist = 100
    par.N = par.N_seq+par.N_dist

    par.freq = 10.
    par.jitter = 2.
    par.onset = 1

    par.dt = .05
    par.tau_m = 10.
    par.v_th = 1.4
    par.tau_x = 2.

    par.T = int(2*(par.Dt*par.N_seq + par.jitter) / (par.dt))

    par.dir_output = '../_results/'

    '-----------------------------------------'

    path = get_dir_results(par)
    if not os.path.exists(path): os.makedirs(path)
    get_hyperparameters(par,path)

    'create log files'
    log = os.path.join(path, 'train.txt')
    with open(log,'w') as train:
        train.write('epoch, loss_train, loss_test \n')
    
    loaddir = ('../_datasets/{}/N_seq_{}_N_dist_{}_Dt_{}/'+
               'freq_{}_jitter_{}_onset_{}/').format(par.name,par.N_seq,par.N_dist,par.Dt,
                                             par.freq,par.jitter,par.onset)
        
    train_data = np.load(loaddir+'x_train.npy')
    test_data = np.load(loaddir+'x_test.npy')
    train_onset = np.load(loaddir+'onsets_train.npy')
    test_onset = np.load(loaddir+'onsets_test.npy')
        
    ## complete online training: one example per batch
    par.train_nb = par.batch
    par.test_nb = par.batch
        
    neuron = NeuronClassNumPy(par)
    neuron.initialize()

    'train'
    trainer = TrainerClass(par,neuron,train_data,test_data,train_onset,test_onset)
    trainer.train(log)

    np.save(path+'loss_train',trainer.losstrainList)
    np.save(path+'loss_test',trainer.losstestList)
    np.save(path+'v',trainer.vList)
    np.save(path+'fr',trainer.frList)
    np.save(path+'z',trainer.zList)
    np.save(path+'onset',trainer.onsetList)
    np.save(path+'w',trainer.w)