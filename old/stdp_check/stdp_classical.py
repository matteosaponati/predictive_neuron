import numpy as np
import sys

from predictive_neuron import models, funs









dir = '/predictive_plasticity/'
maindir = dir+'fig4_stdp/'
sys.path.append(dir), sys.path.append(maindir)

import funs_stdp as funs

'-----------------------------------------------------------------------------'
'CLASSICAL STDP WINDOW'

'model parameters'
p_num = {}
p_num['dt'] = .05
p_num['eta'] = 2e-4
p_num['v_th'] = 2.
p_num['gamma'] = .0

'simulation parameters'
T, epochs = 500, 60
tau_x, A_x = 2, 1

'initial conditions'
w_0_pre = np.array([.001,.11])
w_0_post = np.array([.11,.001])
delay = np.arange(4.,60,.05)

"""
stdp window
inputs:
    1. delay: set of delays between pre and post
    2. T: duration of one epoch
    3. epochs: total number of epochs
    4. w_0_pre: initial weights in pre-post pairing
    5. w_0_post: initial weights in post-pre pairing
    6. A_x: amplitude of input spke
    7. tau_x: time constant of input kernel
    8. p_num: model parameters
output:
    w_pre: weights of sub-threshold input after training pre-post, for all delays
    w_post: weights of sub-threshold input after training post-pre, for all delays
"""
p_num['tau'] = 10.
w_pre_10, w_post_10 = funs.stdp_window(delay,T,epochs,w_0_pre,w_0_post,A_x,tau_x,p_num)
p_num['tau'] = 15.
w_pre_15, w_post_15 = funs.stdp_window(delay,T,epochs,w_0_pre,w_0_post,A_x,tau_x,p_num)
p_num['tau'] = 20.
w_pre_20, w_post_20 = funs.stdp_window(delay,T,epochs,w_0_pre,w_0_post,A_x,tau_x,p_num)

'plots - panel b'
funs.plot_stdp_window(delay,[w_pre_10,w_pre_15,w_pre_20],[w_post_10,w_post_15,w_post_20],w_0_pre[0],w_0_post[1],maindir)