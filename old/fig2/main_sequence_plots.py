"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_sequence_plots.py"
make example plots - Fig 2

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

'---------------------------------------'
'auxiliary functions plots'
def hex_to_rgb(value):
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
def rgb_to_dec(value):
    return [v/256 for v in value]
def get_continuous_cmap(hex_list, float_list=None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

'---------------------------------------'
def plot_w_dynamics(w,par):
    hex_list = ['#FFFAF0','#33A1C9','#7D26CD'] 
    fig = plt.figure(figsize=(4,6), dpi=300)    
    divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
    plt.imshow(np.flipud(w.T),cmap=get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.ylabel('inputs')
    plt.xlabel('epochs')
    plt.xlim(0,2000)
    plt.savefig(par.savedir+'w_norm.png',format='png', dpi=300)
    plt.savefig(par.savedir+'w_norm.pdf',format='pdf', dpi=300)
    plt.close('all')
    return

'---------------------------------------'
def plot_w_dynamics_zoom(w,par):
    hex_list = ['#FFFAF0','#33A1C9','#7D26CD']  
    w_norm = w/(w[0,:]*2)
    fig = plt.figure(figsize=(4,2), dpi=300)    
    divnorm = colors.DivergingNorm(vmin=w_norm.T.min(),vcenter=1, vmax=w_norm.T.max())
    plt.imshow(np.flipud(w_norm[:,:10].T),cmap=get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.ylabel('N')
    plt.xlabel('epochs')
    plt.xlim(0,2000)
    plt.savefig(par.savedir+'w_zoom.png',format='png', dpi=300)
    plt.savefig(par.savedir+'w_zoom.pdf',format='pdf', dpi=300)
    plt.close('all')
    return

'---------------------------------------'
def plot_spk_activity(w,spk,par):
    fig = plt.figure(figsize=(6,6), dpi=300)
    for k,j in zip(spk,range(w.shape[0])):
        plt.scatter([j]*len(k),k,edgecolor='royalblue',s=5)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlim(0,w.shape[0])
    plt.ylim(0,par.T*par.dt)
    plt.xlabel('epochs')
    plt.ylabel('spk times [ms]')
    plt.savefig(par.savedir+'spk_activity.png',format='png', dpi=300)
    plt.savefig(par.savedir+'spk_activity.pdf',format='pdf', dpi=300)
    plt.close('all') 
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
    parser.add_argument('--rep', type=int, default=1)
    
    'input sequence'
    parser.add_argument('--spk_volley',type=str, 
                        choices=['deterministic','random'],default='random',
                        help='type of spike volley')
    parser.add_argument('--Dt', type=int, default=4) 
    parser.add_argument('--N_seq', type=int, default=10)
    parser.add_argument('--N_dist', type=int, default=15)
    parser.add_argument('--offset', type=bool, default=False)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2) 
    
    'neuron model'
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    par = parser.parse_args()
    'additional parameters'
    par.savedir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences_plots/'
    par.loaddir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/sequences/'
#    par.device = "cuda" if torch.cuda.is_available() else "cpu"
    par.device = "cpu"
    par.tau_x = 2.
    
    'set total length of simulation'
    par.T = int((par.Dt*par.N_seq)/(par.dt))
    'set total input'
    par.N = par.N_seq+par.N_dist    
    
    'load data'
    w = np.load(par.loaddir+'w_tau_{}_vth_{}_rep_{}.npy'.format(par.tau_m,par.v_th,par.rep))
    spk = np.load(par.loaddir+'spk_tau_{}_vth_{}_rep_{}.npy'.format(par.tau_m,par.v_th,par.rep),
                  allow_pickle=True)
    
    'plot w dynamics'
    plot_w_dynamics(w,par)    
    'plot evolution of spk times'
    plot_spk_activity(w,spk,par)    
    
    '--------------------'
