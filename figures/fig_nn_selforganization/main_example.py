"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_example.py"
neural network with self-organization lateral connections - get example as from
""

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models_nn, funs

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
def plot_w_matrix(w,par):    
    
    'plot final weight matrix'
    hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
    fig = plt.figure(figsize=(6,6), dpi=300)    
    divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
    plt.imshow(w[-1,:],cmap=get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.ylabel(r'$N_{in}$')
    plt.xlabel(r'$N_{nn}$')
    plt.savefig(par.savedir+'w_nn.png',format='png', dpi=300)
    plt.savefig(par.savedir+'w_nn.pdf',format='pdf', dpi=300)
    plt.close('all')
    return

def plot_w_dynamics(w,par):
    hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
    for n in range(par.nn):
        fig = plt.figure(figsize=(6,6), dpi=300)    
        divnorm = colors.DivergingNorm(vmin=w.T.min(),vcenter=0, vmax=w.T.max())
        plt.imshow(np.flipud(w[:,:,n].T),cmap=get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        plt.colorbar()
        plt.ylabel(r'$N_{in}$')
        plt.xlabel(r'epochs')
        plt.xlim(0,1000)
        plt.savefig(par.savedir+'w_nn_{}.png'.format(n),format='png', dpi=300)
        plt.savefig(par.savedir+'w_nn_{}.pdf'.format(n),format='pdf', dpi=300)
        plt.close('all')
    return

'---------------------------------------'
def plot_spk_activity(w,spk_times,par):
    for n in range(w.shape[-1]):
        fig = plt.figure(figsize=(6,6), dpi=300)
        for k,j in zip(spk_times[n],range(w.shape[0])):
            plt.scatter([j]*len(k[0]),k[0],edgecolor='royalblue',facecolor='none',s=7)
        fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        plt.xlim(0,w.shape[0]-1000)
        plt.ylim(0,par.T*par.dt)
        plt.xlabel('epochs')
        plt.ylabel('spk times [ms]')
        plt.savefig(par.savedir+'spk_n_{}.png'.format(n),format='png', dpi=300)
        plt.savefig(par.savedir+'spk_n_{}.pdf'.format(n),format='pdf', dpi=300)
        plt.close('all') 
    return

'---------------------------------------'
def plot_network_activity(w,par):
    
    'define input sequence'
    timing = [[] for n in range(par.nn)]
    spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
    for n in range(par.nn): 
        for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)
    
    '---------------------------------------'
    'create network dynamics - first epoch'
    x_data = funs.get_sequence_nn_selforg_noise(par,timing)
    network = models_nn.NetworkClass_SelfOrg(par)
    network.w = nn.Parameter(torch.from_numpy(w[0,:])).to(par.device)
    network.state()
    network, v_before, z_before = network(par,network,x_data)
    'create network dynamics - final epoch'
    x_data = funs.get_sequence_nn_selforg_noise(par,timing)
    network = models_nn.NetworkClass_SelfOrg(par)
    network.w = nn.Parameter(torch.from_numpy(w[-1,:])).to(par.device)
    network.state()
    network, v_after, z_after = network(par,network,x_data)
    
    '---------------------------------------'
    'plot network activity'
    colors = ['paleturquoise','lightseagreen','lightblue','dodgerblue','royalblue','mediumblue','mediumslateblue','midnightblue']
    
    'before'
    fig = plt.figure(figsize=(5,3), dpi=300)
    m=1
    for n in range(par.nn):
        plt.eventplot(z_before[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
        m+=1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.ylim(1,par.nn)
    plt.xlim(0,par.T*par.dt)
    plt.xlabel('time [ms]')
    plt.ylabel(r'$N_{nn}$')
    plt.savefig(par.savedir+'spk_before.png',format='png', dpi=300)
    plt.savefig(par.savedir+'spk_before.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    v_spike = v_before[0,:,:].copy()
    v_spike[v_spike>3.5]=9
    fig = plt.figure(figsize=(5,4), dpi=300)
    count = 0
    for n in range(0,par.nn,1):
        plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=colors[count])
        count +=1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.savefig(par.savedir+'v_before.png',format='png', dpi=300)
    plt.savefig(par.savedir+'v_before.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    'after'
    fig = plt.figure(figsize=(5,3), dpi=300)
    m=1
    for n in range(par.nn):
        plt.eventplot(z_after[n][0],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
        m+=1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.ylim(1,par.nn)
    plt.xlim(0,par.T*par.dt)
    plt.xlabel('time [ms]')
    plt.ylabel(r'$N_{nn}$')
    plt.savefig(par.savedir+'spk_after.png',format='png', dpi=300)
    plt.savefig(par.savedir+'spk_after.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    v_spike = v_after[0,:,:].copy()
    v_spike[v_spike>3.5]=9
    fig = plt.figure(figsize=(5,4), dpi=300)
    count = 0
    for n in range(0,par.nn,1):
        plt.plot(np.linspace(0,par.T*par.dt,par.T),v_spike[:,n],linewidth=2,color=colors[count])
        count +=1
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('time [ms]')
    plt.savefig(par.savedir+'v_after.png',format='png', dpi=300)
    plt.savefig(par.savedir+'v_after.pdf',format='pdf', dpi=300)
    plt.close('all') 
    
    return

'---------------------------------------'
def plot_activity_duration(w,spk_times,par):
    duration = []
    for k in range(w.shape[0]):
        if len(spk_times[k][-1][0]) != 0 and len(spk_times[k][0][0]) != 0:
            duration.append(spk_times[k][-1][0][-1]-spk_times[k][0][0][0])
        else: duration.append(spk_times[0][-1][0][0]-spk_times[0][0][0][0])
    
    fig = plt.figure(figsize=(5,6), dpi=300)
    plt.plot(duration,linewidth=2,color='purple')
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.xlabel('epochs')
    plt.ylabel(r'$\Delta t$ [ms]')
    plt.savefig(par.savedir+'duration_activity.png',format='png', dpi=300)
    plt.savefig(par.savedir+'duration_acitivity.pdf',format='pdf', dpi=300)
    plt.close('all') 
    return duration

'---------------------------------------------------------'
'---------------------------------------------------------'

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
                    description="""
                    single neuron trained on spike patterns
                    """
                    )
    'initialization'
    parser.add_argument('--init',type=str, 
                        choices=['classic','trunc_gauss','fixed'],default='fixed',
                        help='type of weights initialization')
    parser.add_argument('--init_mean',type=float, default=0.01)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.08)
    parser.add_argument('--w_0',type=float, default=.08,
                        help='fixed initial condition')
    parser.add_argument('--w_0rec',type=float, default=.0003,
                        help='fixed initial condition')
    'optimizer'
    parser.add_argument('--online', type=bool, default=True,
                        help='set online learning algorithm')
    parser.add_argument('--batch', type=int, default=1,
                        help='number of batches') 
    parser.add_argument('--eta',type=float, default=1e-5,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=1500,
                        help='number of epochs')
    parser.add_argument('--seed', type=int, default=1992)
      
    'setup inputs'
    parser.add_argument('--n_in', type=int, default=2)
    parser.add_argument('--delay', type=int, default=4)
    parser.add_argument('--Dt', type=int, default=2)
    parser.add_argument('--fr_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=.01) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2) 

    'neuron model'
    parser.add_argument('--is_rec', type=bool, default=True,
                        help='set recurrent connections')
    parser.add_argument('--nn', type=int, default=6)
    parser.add_argument('--lateral', type=int, default=2)
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 20.) 
    parser.add_argument('--v_th', type=float, default= 3.)
    parser.add_argument('--dtype', type=str, default=torch.float)
    
    parser.add_argument('--savedir', type=str, default='') 
    
    par = parser.parse_args()
    'additional parameters'
    par.savedir = ''
    par.device = "cpu"
    par.tau_x = 2.
    par.T = int((par.n_in*par.delay+(par.n_in*par.Dt)+80)/par.dt)
    
    '----------'
    'upload data'
    spk_times = np.load(dir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                        par.n_in,par.nn,par.delay,par.Dt,
                                        par.tau_m,par.v_th,par.w_0),allow_pickle=True).tolist()
    w = np.load(dir+'w_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                        par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th,par.w_0))
    '----------'
    
    'plot final w matrix'
    plot_w_matrix(w,par)
    
    'plot w dynamics'
    plot_w_dynamics(w,par)
    
    'plot evolution of spk times'
    plot_spk_activity(w,spk_times,par)
    
    'plot network activity'
    plot_network_activity(w,par)
    
    'plot duration of activity'
    plot_activity_duration(w,spk_times,par)