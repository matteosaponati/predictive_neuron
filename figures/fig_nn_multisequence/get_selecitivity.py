import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rc('axes', axisbelow=True)

#%%


selectivity = np.zeros((par.nn,par.batch,par.epochs))
for b in range(par.batch):
    for n in range(par.nn):
        for e in range(par.epochs):
            if z_out[n][e][b] != []: selectivity[n,b,e] = 1
    
#%%
n = 9
fig = plt.Figure(figsize=(40,10),dpi=150)
plt.subplot(1,par.batch+1,1)
# for b in range(par.batch):
#     plt.plot(selectivity[n,b,:],linewidth=2,label='seq {}'.format(b+1))
plt.imshow(selectivity[n,:],aspect='auto',cmap ='Greys_r',interpolation='none',vmin=0,vmax=1)
plt.colorbar()
# plt.xlim(0,800)
plt.yticks(range(par.batch),range(1,par.batch+1))
plt.xlim(0,par.epochs)
plt.xlabel('epochs')
# plt.ylabel('sequence number')
for b in range(par.batch):    
    plt.subplot(1,par.batch+1,b+2)
    spk = []
    for k in range(par.epochs): spk.append(z_out[n][k][b])
    for k,j in zip(spk,range(par.epochs)):
        plt.scatter([j]*len(k),k,edgecolor='royalblue',facecolor='none',s=7)
    for k in timing[n][b]:
        plt.axhline(y=(k)*par.dt,color='k')
    plt.ylim(0,par.T*par.dt)
    plt.xlabel('epochs')
    plt.xlim(0,par.epochs)
# plt.ylabel('output spike times [ms]')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])

#%%
n = 1
plt.imshow(selectivity[n,:],aspect='auto')

        


        