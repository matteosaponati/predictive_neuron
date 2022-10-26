"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"plot_parameter_sweep.py"

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

tau_m_sweep = np.arange(1.,30.,1.)
tau_x_sweep =  np.arange(1.,30.,1.)/2

savedir = '/Users/saponatim/Desktop/'
loaddir = '/Users/saponatim/Desktop/'
parspace = np.load(loaddir+'parspace_taum_taux.npy')
mask = np.load(loaddir+'mask_taum_taux.npy')

hex_list = ['#FFFAF0','#7D26CD'] 
divnorm = colors.LogNorm(vmin=parspace[:11,:].min(),vmax=parspace[:11,:].max())
fig = plt.figure(figsize=(4,6), dpi=300)
plt.imshow(parspace,cmap=get_continuous_cmap(hex_list),norm=divnorm,aspect='auto')
plt.contour(mask,colors='xkcd:navy blue',linewidths=.5,alpha=.7)
plt.yticks(np.arange(len(tau_m_sweep))[::5],tau_m_sweep[::5][::-1])
plt.xticks(np.arange(len(tau_x_sweep))[::10],tau_x_sweep[::10])
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlim(0,11)
#plt.colorbar()
plt.xlabel(r'$\tau_{x}$')
plt.ylabel(r'$\tau_{m}$')
plt.savefig(savedir+'parameter_space_taum_vth.png',format='png', dpi=300)
plt.savefig(savedir+'parameter_space_taum_vth.pdf',format='pdf', dpi=300)
plt.close('all')
