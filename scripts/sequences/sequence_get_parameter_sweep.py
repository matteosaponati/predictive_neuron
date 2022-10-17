"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"get_parameter_sweep.py"
compute how sequence prediction changes in parameter space

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import numpy as np

'---------------------------------------'

def do_matrix(par):
    
    if par.type == 'taum_taux':
        matrix = np.zeros((len(par.tau_m_sweep),len(par.tau_x_sweep)))
        mask = np.zeros((len(par.tau_m_sweep),len(par.tau_x_sweep)))    
    
        for tm in range(len(par.tau_m_sweep)):
            for tx in range(len(par.tau_x_sweep)):
    
                par.tau_m = par.tau_m_sweep[tm]
                par.tau_x = par.tau_x_sweep[tx]            
                spk = np.load(par.loaddir+'spk_tau_{}_taux_{}.npy'.format(par.tau_m,par.tau_x),allow_pickle=True).tolist()
                if spk[-1][0] != []: matrix[tm,tx] = len(spk[-1])
                if spk[-1][0] < 20: mask[tm,tx] = 1
                
        np.save(par.savedir+'parspace_taum_taux',matrix)
        np.save(par.savedir+'mask_taum_taux',mask)
        
    if par.type == 'taum_vth':
        matrix = np.zeros((len(par.tau_m_sweep),len(par.v_th_sweep)))
        mask = np.zeros((len(par.tau_m_sweep),len(par.v_th_sweep)))    
    
        for tm in range(len(par.tau_m_sweep)):
            for th in range(len(par.v_th_sweep)):
    
                par.tau_m = par.tau_m_sweep[tm]
                par.tau_x = par.v_th_sweep[th]            
                spk = np.load(par.loaddir+'spk_tau_{}_taux_{}.npy'.format(par.tau_m,par.tau_x),allow_pickle=True).tolist()
                if spk[-1][0] != []: matrix[tm,th] = len(spk[-1])
                if spk[-1][0] < 20: mask[tm,th] = 1
                
        np.save(par.savedir+'parspace_taum_vth',matrix)
        np.save(par.savedir+'mask_taum_vth',mask)
    
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
    'sweep'
    parser.add_argument('--type',type=str, 
                        choices=['taum_taux','taum_vth'],default='taum_taux',
    parser.add_argument('--tau_m_sweep', type=list, 
                        default=[])
    parser.add_argument('--tau_x_sweep', type=list, 
                        default=[])
    parser.add_argument('--v_th_sweep', type=list, 
                        default=[])
    
    'neuron model'
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--tau_x', type=float, default= 2.)
     parser.add_argument('--v_th', type=float, default= 2.)
    
    par = parser.parse_args()
    par.loaddir = '/Users/saponatim/Desktop/'
    par.savedir = '/Users/saponatim/Desktop/'
    
    do_matrix(par)
