import argparse
import json
import os
import torch

from utils.funs import get_dir_results

'-------------------------------'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    """
    Arguments:
        --name (str): Type of simulation. Choices: 'sequence', 'multisequence', 'selforg'. Default: 'sequence'.
        --package (str): Implementation package to use. Choices: 'NumPy', 'PyTorch'. Default: 'NumPy'.
        --bound (str): Training algorithm. Choices: 'none', 'hard', 'soft'. Default: 'none'.
        --eta (float): Learning rate. Default: 1e-5.
        --optimizer (str): Optimization algorithm. Choices: 'SGD', 'Adam', 'RMSprop'. Default: 'Adam'.
        --batch (int): Batch size. Default: 10.
        --epochs (int): Number of epochs for training. Default: 100.
        --seed (int): Seed for random number generation. Default: 1992.
        --rep (int): Number of repetitions. Default: 1.
        --init (str): Type of weight initialization. Choices: 'classic', 'random', 'fixed'. Default: 'fixed'.
        --init_mean (float): Mean value for weight initialization. Default: 0.03.
        --init_a (float): Parameter 'a' for weight initialization. Default: 0.
        --init_b (float): Parameter 'b' for weight initialization. Default: 0.8.
        --init_rec (float): Recurrent connection weight initialization. Default: 0.003.
        --sequence (str): Set input sequence type. Choices: 'deterministic', 'random'. Default: 'deterministic'.
        --Dt (int): Delay between input spikes. Default: 2.
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
        --tau_m (float): Membrane time constant. Default: 10.
        --v_th (float): Threshold for membrane potential. Default: 2.
        --tau_x (float): Time constant for synaptic dynamics. Default: 2.
        --nn (int): Number of neurons in the network. Default: 10.
        --device (str): Device for computation. Default: "cpu".
        --dir_output (str): Output directory. Default: "../_results/".
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

    parser.add_argument('--network_type', type=str, default='nearest')
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--delay', type=int, default=2)
    parser.add_argument('--n_afferents', type=int, default=1)
    
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--jitter', type=float, default=2.)
    parser.add_argument('--onset', type=int, default=0)

    'model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 1.4)
    parser.add_argument('--tau_x', type=float, default= 2.)
    parser.add_argument('--nn', type=int, default=10)

    'utils'
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--dir_output', type=str, default='../_results/')
    
    par = parser.parse_args()

    if par.name == 'sequence':

        par.T = int(2*(par.Dt*par.N_seq + par.jitter)/(par.dt))
        par.N = par.N_seq+par.N_dist

    if par.name == 'selforg': 
        
        par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
        if par.network_type == 'nearest': par.N = par.n_in+2
        if par.network_type == 'all': par.N = par.n_in+par.nn

    '------------------------------------------------'
    'set hyperparameters file'
    
    path = get_dir_results(par)
    if not os.path.exists(path): os.makedirs(path)
    
    hyperparameters = vars(par)    
    with open(path+'hyperparameters.json','w') as outfile:
        json.dump(hyperparameters,outfile,indent=4)