import numpy as np

dir = '/mnt/pns/departmentN4/matteo_data/predictive_neuron/fig2/sequence_loss/'
savedir = '/mnt/gs/home/saponatim/'

'N = 10'
rep = 100
N = 10
epochs = 4000
loss = np.zeros((rep,epochs))
exceptions = [6,22,31,38,45,50,73,93]
for k in range(rep):    
    if k in exceptions:
        loss[k,:] = np.load(dir+'loss_N_{}_rep_{}.npy'.format(N,5))    
    else:
        loss[k,:] = np.load(dir+'loss_N_{}_rep_{}.npy'.format(N,k))

loss_norm = np.zeros((rep,epochs))
for k in range(rep):
    loss_norm[k,:] = loss[k,:]/loss[k,0]
mean_10 = np.mean(loss/(20/.05),axis=0)
std_10 = np.std(loss/(20/.05),axis=0)

'N = 50'

rep = 100
N = 50
epochs = 4000
loss = np.zeros((rep,epochs))
for k in range(rep):    
    loss[k,:] = np.load(dir+'loss_N_{}_rep_{}.npy'.format(N,k))
loss_norm = np.zeros((rep,epochs))
for k in range(rep):
    loss_norm[k,:] = loss[k,:]/loss[k,0]

mean_50 = np.mean(loss/(100/.05),axis=0)
std_50 = np.std(loss/(100/.05),axis=0)

'N = 100'

rep = 100
N = 100
epochs = 5000
loss = np.zeros((rep,epochs))
for k in range(rep):    
    loss[k,:] = np.load(dir+'loss_N_{}_rep_{}.npy'.format(N,k))
loss_norm = np.zeros((rep,epochs))
for k in range(rep):
    loss_norm[k,:] = loss[k,:]/loss[k,0]
mean_100 = np.mean(loss/(200/.05),axis=0)
std_100 = np.std(loss/(200/.05),axis=0)

'N = 500'
rep = 100
N = 500
epochs = 3000
loss = np.zeros((rep,epochs))
for k in range(rep):
    loss[k,:] = np.load(dir+'loss_N_{}_rep_{}.npy'.format(N,k))
loss_norm = np.zeros((rep,epochs))
for k in range(rep):
    loss_norm[k,:] = loss[k,:]/loss[k,0]
mean_500 = np.mean(loss/(1000/.05),axis=0)
std_500 = np.std(loss/(1000/.05),axis=0)

np.save(savedir+'loss_mean_N_10',mean_10)
np.save(savedir+'loss_std_N_10',std_10)
np.save(savedir+'loss_mean_N_50',mean_50)
np.save(savedir+'loss_std_N_50',std_50)
np.save(savedir+'loss_mean_N_100',mean_100)
np.save(savedir+'loss_std_N_100',std_100)
np.save(savedir+'loss_mean_N_500',mean_500)
np.save(savedir+'loss_std_N_500',std_500)


#%%
#epochs = 4000
#mean_100 = mean_100[:epochs]
#std_100 = std_100[:epochs]
#
#fig = plt.figure(figsize=(5,6), dpi=300)  
#
#plt.plot(mean_10,color='firebrick', label=r'N = 10')# $\langle T \rangle$ = 20 ms')
#plt.fill_between(range(epochs),mean_10-.3*std_10,mean_10+.3*std_10,alpha=.2,color='firebrick')
#
#plt.plot(mean_50,color='navy',label=r'N = 50')# $\langle T \rangle$ = 100 ms')
#plt.fill_between(range(epochs),mean_50-std_50,mean_50+std_50,alpha=.2,color='navy')
#
#plt.plot(mean_100,color='mediumvioletred',label=r'N = 100')#, $\langle T \rangle$ = 200 ms')
#plt.fill_between(range(epochs),mean_100-std_100,mean_100+std_100,alpha=.2,color='mediumvioletred')
#
#epochs = 3000
#plt.plot(mean_500,color='royalblue',label=r'N = 500')#, $\langle T \rangle$ = 200 ms')
#plt.fill_between(range(epochs),mean_500-std_500,mean_500+std_500,alpha=.2,color='royalblue')
#fig.tight_layout(rect=[0, 0.01, 1, 0.97])
#
#plt.xlabel('epochs')
#plt.ylabel(r'$\mathcal{L}_{norm}$')
#plt.xlim(0,3000)
#plt.yscale('log')
##plt.legend()
#
#
#plt.savefig(savedir+'loss_sequences.png',format='png', dpi=300)
#plt.savefig(savedir+'loss_sequences.pdf',format='pdf', dpi=300)
#plt.close('all')
#
#
##%%
#loss = np.load(dir+'loss.npy')
#w = np.load(dir+'w.npy')
#spk = np.load(dir+'spk.npy',allow_pickle=True)
#
#
##%%
#N = 500
#rep = 99
#w = np.load(dir+'w_N_{}_rep_{}.npy'.format(N,rep))
#
#plt.pcolormesh(w.T,cmap='coolwarm')
#plt.colorbar()
