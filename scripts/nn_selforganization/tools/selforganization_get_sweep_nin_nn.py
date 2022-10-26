import numpy as np
import argparse
import matplotlib.pyplot as plt

def do_matrix(par,nin_sweep,nn_sweep):
    
    savedir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/nn_selforganization_plots/'
    
    matrix_post = np.zeros((len(nin_sweep),len(nn_sweep)))
    matrix_pre = np.zeros((len(nin_sweep),len(nn_sweep)))
    
    for t in range(len(nin_sweep)):
        for v in range(len(nn_sweep)):
            
            'get valuees of tau_m and v_th'
            par.n_in = nin_sweep[t]
            par.nn = nn_sweep[v]
            
            T = par.nn*4 + (par.n_in*2)
            spk_times = np.load(par.dir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                    par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th,par.w_0),allow_pickle=True).tolist()
            
#            for n in range(par.nn):
            if len(spk_times[-1][0][0]) != 0 and len(spk_times[0][0][0]) != 0:
                pre = spk_times[-1][0][0][-1]-spk_times[0][0][0][0]
            else: pre = 0
            if len(spk_times[-1][-1][0]) != 0 and len(spk_times[0][-1][0]) != 0:
                post = spk_times[-1][-1][0][-1]-spk_times[0][-1][0][0]
            else: post = 0
            
            matrix_post[t,v] = post/T
            matrix_pre[t,v] = pre
                
    
    np.save(savedir+'matrix_sweep_tau_{}_vth_{}_pre'.format(par.tau_m,par.v_th),matrix_pre)
    np.save(savedir+'matrix_sweep_tau_{}_vth_{}_post'.format(par.tau_m,par.v_th),matrix_post)
    
    'make figure'
    fig = plt.figure(figsize=(7,6), dpi=300)
    plt.imshow(np.flipud(matrix_pre),aspect='auto')
    plt.yticks(range(len(nin_sweep)),nin_sweep[::-1])
    plt.xticks(range(len(nin_sweep)),nn_sweep)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.xlabel(r'$nn$')
    plt.ylabel(r'$nin$')
    plt.savefig(savedir+'matrix_sweep_tau_{}_vth_{}_pre.png'.format(par.tau_m,par.v_th),format='png', dpi=300)
    plt.close('all')
    
    fig = plt.figure(figsize=(7,6), dpi=300)
    plt.imshow(np.flipud(matrix_post),aspect='auto')
    plt.yticks(range(len(nin_sweep)),nin_sweep[::-1])
    plt.xticks(range(len(nin_sweep)),nn_sweep)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.xlabel(r'$nn$')
    plt.ylabel(r'$nin$')
    plt.savefig(savedir+'matrix_sweep_tau_{}_vth_{}_post.png'.format(par.tau_m,par.v_th),format='png', dpi=300)
    plt.close('all')
    
#    fig = plt.figure(figsize=(7,6), dpi=300)
#    plt.imshow(np.flipud((matrix_post-matrix_pre)),aspect='auto')
#    plt.yticks(range(len(nin_sweep)),nin_sweep[::-1])
#    plt.xticks(range(len(nin_sweep)),nn_sweep)
#    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
#    plt.colorbar()
#    plt.xlabel(r'$nn$')
#    plt.ylabel(r'$nin$')
#    plt.savefig(savedir+'matrix_sweep_tau_{}_vth_{}.png'.format(par.tau_m,par.v_th),format='png', dpi=300)
#    plt.close('all')

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description=
                                     """
                                     nn selforganization lateral
                                     """)
    par = parser.parse_args()
    par.delay = 4
    par.Dt = 2
    par.w_0 = .02
    par.dir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/nn_selforganization_parameter_sweep/'
    
    par.tau_m = 15.
    par.v_th = 3.5
    
    nin_sweep = [x for x in range(2,31,1)]
    nn_sweep = [x for x in range(2,21,1)]
    
    print('matrix tau_m {} v_th {} '.format(
                        par.tau_m,par.v_th))
    do_matrix(par,nin_sweep,nn_sweep)
