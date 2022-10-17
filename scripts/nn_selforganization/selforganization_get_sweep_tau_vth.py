import numpy as np
import argparse
import matplotlib.pyplot as plt

def do_matrix(par,tau_sweep,vth_sweep):
    
    savedir = '/mnt/hpc/departmentN4/matteo_data/predictive_neuron/nn_selforganization_plots/'
    
    matrix_post = np.zeros((len(tau_sweep),len(vth_sweep)))
    matrix_pre = np.zeros((len(tau_sweep),len(vth_sweep)))
    
    for t in range(len(tau_sweep)):
        for v in range(len(vth_sweep)):
            
            'get valuees of tau_m and v_th'
            par.tau_m = tau_sweep[t]
            par.v_th = vth_sweep[v]
            
            spk_times = np.load(par.dir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
                                    par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th,par.w_0),allow_pickle=True).tolist()
            
            if len(spk_times[-1][0][0]) != 0 and len(spk_times[0][0][0]) != 0:
                pre = spk_times[-1][0][0][-1]-spk_times[0][0][0][0]
            else: pre = 0
            if len(spk_times[-1][-1][0]) != 0 and len(spk_times[0][-1][0]) != 0:
                post = spk_times[-1][-1][0][-1]-spk_times[0][-1][0][0]
            else: post = 0
            
            matrix_post[t,v] = post
            matrix_pre[t,v] = pre
    
    np.save(savedir+'matrix_sweep_nin_{}_nn_{}_pre'.format(par.n_in,par.nn),matrix_pre)
    np.save(savedir+'matrix_sweep_nin_{}_nn_{}_post'.format(par.n_in,par.nn),matrix_post)
    
    'make figure'
    fig = plt.figure(figsize=(7,6), dpi=300)
    plt.imshow(np.flipud(matrix_pre),aspect='auto')
    plt.yticks(range(len(tau_sweep)),tau_sweep[::-1])
    plt.xticks(range(len(vth_sweep)),vth_sweep)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.xlabel(r'$tau_m$')
    plt.ylabel(r'$v_{th}$')
    plt.savefig(savedir+'matrix_sweep_nin_{}_nn_{}_pre'.format(par.n_in,par.nn),format='png', dpi=300)
    plt.close('all')
    
    fig = plt.figure(figsize=(7,6), dpi=300)
    plt.imshow(np.flipud(matrix_post),aspect='auto')
    plt.yticks(range(len(tau_sweep)),tau_sweep[::-1])
    plt.xticks(range(len(vth_sweep)),vth_sweep)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.xlabel(r'$tau_m$')
    plt.ylabel(r'$v_{th}$')
    plt.savefig(savedir+'matrix_sweep_nin_{}_nn_{}_post'.format(par.n_in,par.nn),format='png', dpi=300)
    plt.close('all')
    
    fig = plt.figure(figsize=(7,6), dpi=300)
    plt.imshow(np.flipud((matrix_post-matrix_pre)/matrix_pre),aspect='auto')
    plt.yticks(range(len(tau_sweep)),tau_sweep[::-1])
    plt.xticks(range(len(vth_sweep)),vth_sweep)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.colorbar()
    plt.xlabel(r'$tau_m$')
    plt.ylabel(r'$v_{th}$')
    plt.savefig(savedir+'matrix_sweep_nin_{}_nn_{}'.format(par.n_in,par.nn),format='png', dpi=300)
    plt.close('all')

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
    
    tau_sweep = [5.,10.,15.,20.]
    vth_sweep = [1.,1.5,2.,2.5,3.,3.5]
    
    nin_sweep = [x for x in range(2,26,4)]
    nn_sweep = [x for x in range(2,22,4)]
    
    
    for n_in in nin_sweep:
        for nn in nn_sweep:
            
            par.n_in = n_in
            par.nn = nn
            
            print('matrix nin {} nn {} '.format(
                                n_in,nn))
            
            do_matrix(par,tau_sweep,vth_sweep)
