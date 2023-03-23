import argparse
import os

from utils.funs import get_dir_results, get_hyperparameters

'-------------------------------'

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name',type=str, 
                        choices = ['sequence','multisequence','selforg'],
                        default = 'sequence',
                        help = 'type of simulation')
    
    parser.add_argument('--package',type=str, 
                        choices = ['NumPy','PyTorch'],
                        default = 'NumPy',
                        help = 'which implementation to use')
    
    'training algorithm'    
    parser.add_argument('--bound',type=str,
                        choices=['none','hard','soft'],default='none')
    parser.add_argument('--eta',type=float, default=1e-5)
    parser.add_argument('--optimizer',type=str,
                        choices = ['SGD','Adam','RMSprop'],default='Adam')
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--rep', type=int, default=1)
    
    'initialization'
    parser.add_argument('--init',type=str, 
                        choices=['classic','random','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--init_mean',type=float, default=0.03)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.8)
    parser.add_argument('--init_rec',type=float, default=.003)
    
    'set input sequence'
    parser.add_argument('--sequence',type=str, 
                        choices=['deterministic','random'],default='deterministic')
    parser.add_argument('--Dt', type=int, default=2) 
    
    parser.add_argument('--N_seq', type=int, default=100)
    parser.add_argument('--N_dist', type=int, default=100)

    parser.add_argument('--network_type', type=str,
                        choices = ['nearest','all','random'],default='nearest')
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--delay', type=int, default=2)
    
    parser.add_argument('--freq', type=float, default=10.)
    parser.add_argument('--jitter', type=float, default=2.)
    parser.add_argument('--onset', type=int, default=1)

    'model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--tau_x', type=float, default= 2.)

    parser.add_argument('--nn', type=int, default=10)
  
    parser.add_argument('--dir_output', type=str, default='../_results/')
    
    par = parser.parse_args()

    if par.name == 'sequence':

        par.T = int(2*(par.Dt*par.N_seq + par.jitter) / (par.dt))
        par.N = par.N_seq+par.N_dist
        if par.onset == 1: par.onset = par.T // 2

    if par.name == 'selforg': 
        
        if par.network_type == 'nearest': 
            
            par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
            par.N = par.n_in+2

        if par.network_type == 'all': 
            
            par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
            par.N = par.n_in+par.nn

        if par.network_type == 'random':

            par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                        par.jitter + 80)/(par.dt))
            par.N_in = par.n_in*par.nn
            par.N = par.N_in+par.nn

    '------------------------------------------------'
    'set hyperparameters file'
    
    path = get_dir_results(par)
    if not os.path.exists(path): os.makedirs(path)
    get_hyperparameters(par,path)
    
    '------------------------------------------------'
    
    if par.name == 'sequence':
        from scripts.main_sequence import main

    if par.name == 'selforg':
        from scripts.main_selforg import main

    main(path)    