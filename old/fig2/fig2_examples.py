import torch
import numpy as np
import tools

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    
    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['online','SGD','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--hardbound',type=str,default='False',
                        help='set hard lower bound for parameters')
    parser.add_argument('--init',type=str, 
                        choices=['classic','trunc_gauss','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--init_mean',type=float, default=0.05)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.1)
    parser.add_argument('--w_0',type=float, default=.03,
                        help='fixed initial condition')
    parser.add_argument('--eta',type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=3000,
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches')
    parser.add_argument('--rep', type=int, default=1)   
    
    'input'
    parser.add_argument('--input',type=str, 
                        choices=['sequence','pattern'],default='sequence',
                        help='type of input spike pattern')
    parser.add_argument('--N', type=int, default=100) 
    
    parser.add_argument('--spk_volley',type=str, 
                        choices=['deterministic','random'],default='random',
                        help='type of spike volley')
    parser.add_argument('--Dt', type=int, default=4) 
    
    parser.add_argument('--T_pattern', type=int, default=100)
    parser.add_argument('--freq_pattern', type=float, default=.01) 
    
    'noise'
    parser.add_argument('--fr_noise', type=str, default='False')
    parser.add_argument('--freq', type=float, default=.01) 
    parser.add_argument('--jitter_noise', type=str, default='False')
    parser.add_argument('--jitter', type=int, default=1) 
    parser.add_argument('--offset', type=str, default='False') 
    
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    
    'additional parameters'
    if par.input == 'sequence': par.T = int((par.Dt*par.N*2)/(par.dt))
    if par.input == 'pattern': par.T = int(par.T_pattern*2/par.dt)
    par.device = "cpu"
    par.tau_x = 2.
    
    loss, w, v, spk = tools.train(par)
    
    #np.save('w_{}_offset_{}'.format(par.input,par.offset),w)
    #np.save('spk_{}_offset_{}'.format(par.input,par.offset),spk)
    np.save('w_{}'.format(par.input),w)
    np.save('spk_{}'.format(par.input),spk)  
    np.save('v_{}'.format(par.input),v)  
    np.save('loss_{}'.format(par.input),loss)  
    
    'plots'
#    tools.plot_w_spk(par,w,spk)
#    
#    tools.sequence_example(par,savedir)
#    tools.plot_w_spk(par,w,spk,savedir)
#    tools.density_example(par,timing,savedir)
#    tools.plot_fr_average(par,savedir)

