"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"sequence_compute_metrics.py"
compute metrics for sequence prediction - Fig 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

'---------------------------------------'

def compute_metrics(loss,v,fr,par):
    
    loss_norm = loss/(loss[:,0,None])
    v_norm = v/(v[:,0,None])
    loss_mean = loss_norm.mean(axis=0)
    loss_std = loss_norm.std(axis=0)
    v_mean = v_norm.mean(axis=0)
    v_std = v_norm.std(axis=0)
    
    np.save(par.savedir+'loss_mean_N_{}'.format(par.N_seq),loss_mean)
    np.save(par.savedir+'loss_std_N_{}'.format(par.N_seq),loss_std)
    np.save(par.savedir+'v_mean_N_{}'.format(par.N_seq),v_mean)
    np.save(par.savedir+'v_std_N_{}'.format(par.N_seq),v_std)
    np.save(par.savedir+'fr_mean_N_{}'.format(par.N_seq),fr.mean(axis=0))
    np.save(par.savedir+'fr_std_N_{}'.format(par.N_seq),fr.std(axis=0))
    
    return

'---------------------------------------'
'---------------------------------------'

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    single neuron trained on sequences
                    """
                    )
    
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches')
    parser.add_argument('--rep_tot', type=int, default=100)
    
    'input sequence'
    parser.add_argument('--spk_volley',type=str, 
                        choices=['deterministic','random'],default='random',
                        help='type of spike volley')
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N_seq', type=int, default=10)
    parser.add_argument('--N_dist', type=int, default=10)
    parser.add_argument('--offset', type=bool, default=False)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2)
    parser.add_argument('--epochs', type=int, default=2000)  
    
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 1.5)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    'additional parameters'
    par.savedir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences_N_{}/'.format(par.N_seq)
    par.loaddir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences_N_{}/'.format(par.N_seq)
#    par.device = "cuda" if torch.cuda.is_available() else "cpu"
    par.device = "cpu"
    par.tau_x = 2.
    
    'set total length of simulation'
    par.T = int((par.Dt*par.N_seq)/(par.dt))
    'set total input'
    par.N = par.N_seq+par.N_dist    
    
    'load data'
    v = np.zeros((par.rep_tot,par.epochs))
    loss = np.zeros((par.rep_tot,par.epochs))
    fr = np.zeros((par.rep_tot,par.epochs))
    
    for k in range(par.rep_tot):
        v[k,:] = np.load(par.loaddir+'v_tau_{}_vth_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,k)).squeeze(axis=1).sum(axis=1)
        loss[k,:] = np.load(par.loaddir+'loss_tau_{}_vth_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,k))
        spk = np.load(par.loaddir+'spk_tau_{}_vth_{}_rep_{}.npy'.format(
                            par.tau_m,par.v_th,k),allow_pickle=True).tolist()
        fr[k,:] = np.array([(len(spk[k])/par.T)*(1e3/par.T)*1e3 for k in range(len(spk))])

    compute_metrics(loss,v,fr,par)    
    
    '--------------------'