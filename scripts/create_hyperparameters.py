import argparse
import json
import os

from utils.funs import get_dir_results

'-------------------------------'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
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

    parser.add_argument('--network_type', type=str, default='nearest')
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--delay', type=int, default=2)
    
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--jitter', type=float, default=2.)
    parser.add_argument('--onset', type=int, default=0)

    'model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 1.4)
    parser.add_argument('--tau_x', type=float, default= 2.)

    parser.add_argument('--nn', type=int, default=10)
  
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