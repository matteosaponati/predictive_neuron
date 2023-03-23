
import numpy as np
import os
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--name',type=str, 
                        choices = ['sequence','multisequence','selforg'],
                        default = 'sequence',
                        help = 'type of simulation')
    
    parser.add_argument('--package',type=str, 
                        choices = ['NumPy','PyTorch'],
                        default = 'NumPy',
                        help = 'which implementation to use')
    
    parser.add_argument('--mode', type=str, default='train')

    'set input sequence'
    parser.add_argument('--sequence',type=str, 
                        choices=['deterministic','random'],default='deterministic')
    parser.add_argument('--Dt', type=int, default=2) 

    parser.add_argument('--batch', type=int, default=20)     
    
    parser.add_argument('--N_seq', type=int, default=100)
    parser.add_argument('--N_dist', type=int, default=100)

    parser.add_argument('--network_type', type=str,
                        choices = ['nearest','all','random'],default='nearest')
    parser.add_argument('--n_in', type=int, default=8)
    parser.add_argument('--delay', type=int, default=4)
    
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--jitter', type=float, default=2.)
    parser.add_argument('--onset', type=int, default=1)

    'model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_x', type=float, default= 2.)

    parser.add_argument('--nn', type=int, default=10)
  
    par = parser.parse_args()
    
    '------------------------------------------------'

    if par.name == 'sequence':

        par.T = int(2*(par.Dt*par.N_seq + par.jitter) / (par.dt))
        par.N = par.N_seq+par.N_dist
        if par.onset == 1: par.onset = par.T // 2

        from utils.data import get_spike_times, get_dataset_sequence

        spk_times = get_spike_times(par)
        x,onsets = get_dataset_sequence(par,spk_times)

        savedir = '../_datasets/{}/N_seq_{}_N_dist_{}_Dt_{}/'.format(par.name,par.N_seq,par.N_dist,par.Dt) + \
                    'freq_{}_jitter_{}_onset_{}/'.format(par.freq,par.jitter,par.onset)
        if not os.path.exists(savedir): os.makedirs(savedir)

        np.save(savedir+'x_{}'.format(par.mode),x)
        np.save(savedir+'onsets_{}'.format(par.mode),onsets)

    if par.name == 'selforg': 
        
        if par.network_type == 'random':

            par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                        par.jitter + 80)/(par.dt))
            par.N_in = par.n_in*par.nn

            from utils.data import get_spike_times, get_dataset_random

            spk_times = get_spike_times(par)
            x = get_dataset_random(par,spk_times)

            savedir = '../_datasets/{}/{}/n_in_{}_nn_{}_Dt_{}/'.format(par.name,par.network_type,par.n_in,par.nn,par.Dt) + \
                    'freq_{}_jitter_{}/'.format(par.freq,par.jitter)
            if not os.path.exists(savedir): os.makedirs(savedir)

            np.save(savedir+'x_{}'.format(par.mode),x)

        else: 

            par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                        par.jitter + 80)/(par.dt))
        
            from utils.data import get_spike_times, get_dataset_selforg

            spk_times = get_spike_times(par)
            x = get_dataset_selforg(par,spk_times)

            savedir = '../_datasets/{}/{}/n_in_{}_nn_{}_delay_{}_Dt_{}/'.format(par.name,par.network_type,par.n_in,par.nn,par.delay,par.Dt) + \
                    'freq_{}_jitter_{}/'.format(par.freq,par.jitter)
            if not os.path.exists(savedir): os.makedirs(savedir)

            np.save(savedir+'x_{}'.format(par.mode),x)