import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import matplotlib.colors as colors
plt.rc('axes', axisbelow=True)

savedir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/sequences/'

'plots'
def plot_convergence_s(spk_tot,timing,epochs,savedir):
    c=['mediumvioletred','mediumslateblue','lightseagreen','salmon']
    fig = plt.figure(figsize=(6,6), dpi=300)
    for s in range(len(spk_tot)):
        for k,j in zip(spk_tot[s][0],range(epochs)):
            plt.scatter([j]*len(k),k,c=c[s],s=7)
    plt.ylabel(r'output spikes (s) [ms]')
    for k in timing:
        plt.axhline(y=k,color='k',linewidth=1.5,linestyle='dashed')
    plt.xlabel(r'epochs')
    plt.xlim(0,epochs)
    plt.ylim(0,10)
    plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(savedir+'s_convergence.png',format='png', dpi=300)
    plt.savefig(savedir+'s_convergence.pdf',format='pdf', dpi=300)
    plt.close('all')
    return

def plot_convergence_w(w,timing,epochs,savedir):
    c=['mediumvioletred','mediumslateblue','lightseagreen','salmon']
    fig = plt.figure(figsize=(6,6), dpi=300)
    for s in range(len(w)):
        plt.plot(w[s][0,:],color=c[s],linewidth=2)
        plt.plot(w[s][1,:],color=c[s],linewidth=2,linestyle='dashed')
    plt.xlabel(r'epochs')
    plt.ylabel(r'synaptic weights $\vec{w}$')
    plt.xlim(0,epochs)
    plt.ylim(bottom=0)
    plt.grid(True,which='both',axis='x',color='darkgrey',linewidth=.7)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    plt.savefig(savedir+'w_convergence.png',format='png', dpi=300)
    plt.savefig(savedir+'w_convergence.pdf',format='pdf', dpi=300)
    plt.close('all')
    return

#%%

for k,j in zip(spk,range(1500)):
    plt.scatter([j]*len(k),k,c='mediumvioletred',s=7)
for k in np.linspace(2,2*100,100):
    plt.axhline(y=k,color='k',linewidth=1.5,linestyle='dashed')
plt.xlabel(r'epochs')
#plt.xlim(0,3000)