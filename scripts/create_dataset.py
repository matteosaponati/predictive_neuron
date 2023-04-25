
import numpy as np
import os
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    """
    Arguments:
        --name (str): Type of simulation. Choices: 'sequence', 'multisequence', 'selforg'. Default: 'sequence'.
        --package (str): Implementation package to use. Choices: 'NumPy', 'PyTorch'. Default: 'NumPy'.
        --mode (str): Type of input. Choices: 'train', 'test'. Default: 'train'.
        --sequence (str): Set input sequence type. Choices: 'deterministic', 'random'. Default: 'deterministic'.
        --Dt (int): Delay between input spikes. Default: 2.
        --batch (int): Batch size. Default: 20.
        --N_seq (int): Number of pre-syn neurons in the sequence. Default: 100.
        --N_dist (int): Number of pre-syn neurons as distractors. Default: 100.
        --network_type (str): Type of network connection. Choices: 'nearest', 'all', 'random'. Default: 'nearest'.
        --n_in (int): Number of input neurons. Default: 100.
        --delay (int): Delay between input subsets. Default: 2.
        --n_afferents (int): Number of afferent neurons. Default: 1.
        --freq (float): Frequency of backgrounf firing in input spikes. Default: 10.
        --jitter (float): Jitter in input spikes. Default: 2.
        --onset (int): Onset time of input spikes. Default: 1.
        --dt (float): Time step for simulations. Default: 0.05.
        --tau_x (float): Time constant for synaptic dynamics. Default: 2.
        --nn (int): Number of neurons in the network. Default: 10.
    """

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
    parser.add_argument('--n_afferents', type=int, default=1)
    
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

        from utils.data import get_spike_times, get_dataset_sequence

        spk_times = get_spike_times(par)
        x,onsets = get_dataset_sequence(par,spk_times)
    
        savedir = '../_datasets/{}/N_seq_{}_N_dist_{}_Dt_{}/'.format(par.name,par.N_seq,par.N_dist,par.Dt) + \
                    'freq_{}_jitter_{}_onset_{}/'.format(par.freq,par.jitter,par.onset)
        if not os.path.exists(savedir): os.makedirs(savedir)

        np.save(savedir+'x_{}'.format(par.mode),x)
        np.save(savedir+'onsets_{}'.format(par.mode),onsets)

    if par.name == 'selforg': 

        par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                        par.jitter + 80)/(par.dt))
        
        from utils.data import get_spike_times

        if par.network_type == 'random': 
            from utils.data import get_dataset_random as get_dataset
            par.N_in = par.n_in*par.nn
        
        else:
            from utils.data import get_dataset_selforg as get_dataset        

        spk_times = get_spike_times(par)
        x = get_dataset(par,spk_times)

        savedir = '../_datasets/{}/{}/n_in_{}_nn_{}_Dt_{}/'.format(par.name,par.network_type,par.n_in,par.nn,par.Dt) + \
                    'freq_{}_jitter_{}/'.format(par.freq,par.jitter)
        if not os.path.exists(savedir): os.makedirs(savedir)

        np.save(savedir+'x_{}'.format(par.mode),x)

    if par.name == 'multisequence':

        par.T = int((par.Dt*par.N_seq + par.n_afferents*par.delay + 
                     par.jitter)/(par.dt))
        par.N = par.N_seq*par.n_afferents 

        from utils.data import get_spike_times, get_dataset_multisequence

        spk_times = get_spike_times(par)
        x = get_dataset_multisequence(par,spk_times)
    
        savedir = '../_datasets/{}/N_seq_{}_n_afferents_{}_Dt_{}_delay_{}/'.format(par.name,par.N_seq,
                                                                                   par.n_afferents,par.Dt,par.delay) + \
                    'freq_{}_jitter_{}/'.format(par.freq,par.jitter)
        if not os.path.exists(savedir): os.makedirs(savedir)

        np.save(savedir+'x_{}'.format(par.mode),x)