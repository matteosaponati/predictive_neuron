"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"plot_supp_parameter_sweep.py"
create example input for self-organizing neural network

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""
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

loaddir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/nn_selforganization_plots/'
savedir = '/mnt/gs/home/saponatim/'
tau_m = 15.
v_th = 3.5

nin_sweep = [x for x in range(2,26,4)]
nn_sweep = [x for x in range(2,22,4)]

'get data'
parspace_pre = np.load(loaddir+'matrix_sweep_tau_{}_vth_{}_pre.npy'.format(tau_m,v_th))
parspace_post = np.load(loaddir+'matrix_sweep_tau_{}_vth_{}_post.npy'.format(tau_m,v_th))

'make figure'
matrix = (parspace_post-parspace_pre)
hex_list = hex_list = ['#33A1C9','#FFFAF0','#7D26CD']
divnorm = colors.DivergingNorm(vmin=matrix.T.min(),vcenter=0, vmax=matrix.T.max())
fig = plt.figure(figsize=(7,6), dpi=300)
plt.imshow(np.flipud(matrix),cmap=get_continuous_cmap(hex_list), norm=divnorm,aspect='auto')
plt.yticks(range(len(nin_sweep)),nin_sweep[::-1])
plt.xticks(range(len(nin_sweep)),nn_sweep)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.colorbar()
plt.xlabel(r'$nn$')
plt.ylabel(r'$nin$')
plt.savefig(savedir+'parameter_space.png',format='png', dpi=300)
plt.savefig(savedir+'parameter_space.pdf',format='pdf', dpi=300)
plt.close('all')