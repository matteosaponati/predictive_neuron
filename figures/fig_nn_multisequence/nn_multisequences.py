import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

import torch.nn.functional as F

#%%

def get_sequence_nn(par,timing):
    
    x_data  = []
    for n in range(par.nn):
        
        x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
        for b in range(par.batch):
             x[b,timing[n][b],range(par.n_in)] = 1
        'synaptic time constant'
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
        x_data.append(x.permute(0,2,1))
        
    return torch.stack(x_data,dim=3)

#%%

class NetworkClass(nn.Module):
    """
    NETWORK MODEL
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        super(NetworkClass,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        
        self.w = nn.Parameter(torch.empty((self.par.n_in,self.par.nn)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in))
        
        if self.par.is_rec == 'True':
            w_rec = np.random.randn(self.par.nn,self.par.nn)/np.sqrt(self.par.nn)
            w_rec = np.where(np.eye(self.par.nn)>0,np.zeros_like(w_rec),w_rec)
            self.wrec = nn.Parameter(torch.as_tensor(w_rec,dtype=self.par.dtype).to(self.par.device))
        
    def state(self):
        """initialization of network state"""
        
        self.v = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        
        self.p = torch.zeros(self.par.batch,self.par.n_in,self.par.nn).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.n_in,self.par.nn).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.n_in,self.par.nn).to(self.par.device)    
        
    def __call__(self,x):
        
        'update membrane voltages'
        for b in range(self.par.batch):
            self.v[b,:] = self.alpha*self.v[b,:] + torch.sum(x[b,:]*self.w,dim=0) \
                     - self.par.v_th*self.z[b,:].detach()
                     
        if self.par.is_rec == True: 
            self.z_out = self.beta*self.z_out + self.z.detach()
            self.v += self.z_out.detach()@self.wrec
        
        'update output spikes'
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z[self.v-self.par.v_th>0] = 1
        
    def backward_online(self,x):
        """
        online evaluation of the gradient:
            - compute the local prediction error 
            - compute the local component of the gradient
            - update the pre-synaptic traces
        """
        
        x_hat = torch.zeros(self.par.batch,self.par.n_in,self.par.nn)
        for b in range(self.par.batch):
            x_hat[b,:] = self.w*self.v[b,:]
            self.epsilon[b,:] = x[b,:] - x_hat[b,:]
            self.grad[b,:] = -(self.v[b,:]*self.epsilon[b,:] \
                             + torch.sum(self.w*self.epsilon[0,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x
        
    def update_online(self):

        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))

#%%
        
'----------------'
def forward(par,network,x_data):
    
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    
    for t in range(par.T):    

        v.append(network.v)         

        if par.online == True: 
            with torch.no_grad():
                network.backward_online(x_data[:,t])
                network.update_online()              
        network(x_data[:,t]) 
        
        for n in range(par.nn):
            for b in range(par.batch):
                if network.z[b,n] != 0: z[n][b].append(t*par.dt)  
        
    return network, torch.stack(v,dim=1), z
'----------------'

#%%
par = types.SimpleNamespace()

'architecture'
par.n_in = 10
par.nn = 5
par.T = 1000
par.batch = 2
par.epochs = 300
par.device = 'cpu'
par.dtype = torch.float

'model parameters'
par.eta = 1e-5
par.dt = .05
par.tau_m = 10
par.v_th = 2.
par.tau_x = 2.

par.is_rec = True
par.online = True

'set inputs'
par.Dt = 4
timing = [[] for n in range(par.nn)]


#%%
'setup inputs'
spk_times = []
for b in range(par.batch):
    spk_times.append(np.random.permutation(np.arange(0,par.Dt*par.n_in,par.Dt)/par.dt))
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b])

x_data = get_sequence_nn(par,timing)

#%%

plt.imshow(x_data[0,:,:,1].T,aspect='auto')

#%%
'set model'
network = NetworkClass(par)
par.w_0 = .05
par.w_0rec = -.05
network.w = nn.Parameter(torch.FloatTensor(par.n_in,par.nn).uniform_(0.,par.w_0))

#network.w = nn.Parameter(w0,dtype=par.dtype).to(par.device)

if par.is_rec == True: 
    w_rec=  par.w_0rec*np.ones((par.nn,par.nn))
    w_rec = np.where(np.eye(par.nn)>0,np.zeros_like(w_rec),w_rec)
    network.wrec = torch.as_tensor(w_rec,dtype=par.dtype).to(par.device)
        
#%%

'setup optimizer'
if par.online != True:   
    optimizer = torch.optim.Adam(network.parameters(),
                                 lr=par.eta,betas=(.9,.999))
#    optimizer = torch.optim.SGD(network.parameters(),lr=par.eta)
    
#%%
    

'setup optimization'
loss_fn = nn.MSELoss(reduction='sum')

w = np.zeros((par.epochs,par.n_in,par.nn))
wrec = np.zeros((par.epochs,par.nn,par.nn))
E = [[] for n in range(par.nn)]
z_out = [[] for n in range(par.nn)]

'training'
for e in range(par.epochs):
        
    network.state()
    network, v, z = forward(par,network,x_data)
    
    x_hat = torch.einsum("btn,jn->btjn",v,network.w)
    
    lossList = []
    for n in range(par.nn):  
        loss = loss_fn(x_hat[:,:,:,n],x_data[:,:,:,n])
        lossList.append(loss)       
    for n in range(par.nn): 
        lossList[n].backward(retain_graph = True)
        E[n].append(lossList[n].item())

    if par.online != True:
        optimizer.step()
        
    w[e,:,:] = network.w.detach().numpy()
    
    w[e,:,:] = network.w.detach().numpy()
    if par.is_rec == True: wrec[e,:,:] = network.wrec.detach().numpy()
    for n in range(par.nn):
        z_out[n].append(z[n])
    
    if e%50 == 0: print(e)
        
#%%


"""
PLOTS
"""


for n in range(par.nn):
    plt.subplot(par.nn,1,n+1)
    for k in range(w.shape[1]):
        plt.plot(w[:,k,n],linewidth=2,label='neuron {}'.format(k+1))
plt.legend()

#%%

'WEIGHT DISTRIBUTION'
fig = plt.figure(figsize=(4,12))
for n in range(par.nn):
    plt.subplot(par.nn,1,n+1)
    plt.xticks([])
    plt.bar(range(1,par.n_in+1),w[-1,:,n],linewidth=2)
plt.xticks(range(1,par.n_in+1))
fig.tight_layout(rect=[0, 0.01, 1, 0.97])

#%%
col = ['navy','mediumvioletred','royalblue','limegreen','orange','skyblue','pink','black']
'FIRING PATTERN'
fig = plt.figure(figsize=(12,12))
for b in range(par.batch):
    plt.subplot(par.batch,par.nn+1,(par.nn+1)*b+1)
    print(par.nn*b+1)
    plt.yticks([])
    plt.xticks(np.arange(0,par.n_in),np.arange(1,par.n_in+1))
    plt.imshow(torch.flip(x_data[b,:,:,0],dims=(0,1)),aspect='auto')

for b in range(par.batch):
    for n in range(par.nn):
        
        plt.subplot(par.batch,par.nn+1,(par.nn+1)*b+n+2)
        plt.xlim(0,par.epochs)
        plt.ylim(0,par.Dt*par.n_in+10)
        'inputs'
        for j in range(par.n_in): plt.axhline(y=j*par.Dt,color='k',linewidth=2,linestyle='dashed')
        'network firing'
        for k in range(len(z_out[n])):
            if z_out[n][k][b] != []:
                plt.scatter([k]*len(z_out[n][k][b]),z_out[n][k][b],s=7,c=col[n])
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
        
#%%  

fig = plt.figure(figsize=(12,12))









    
#%%
for b in range(par.batch):
    plt.subplot(par.batch,1,b+1)
    plt.plot(np.arange(0,par.T*par.dt,par.dt),v_out[-1][b,:,0].detach().numpy(),linewidth=2,color='purple')
    plt.plot(np.arange(0,par.T*par.dt,par.dt),v_out[-1][b,:,1].detach().numpy(),linewidth=2,color='navy')
    plt.axhline(y=par.v_th,color='k')
    plt.xlabel('time [ms]')
    plt.ylabel('v')

#%%

for b in range(par.batch):
    plt.subplot(par.batch,1,b+1)
    plt.axhline(y=timing[0][0][0]*par.dt,color='k',linewidth=2,linestyle='dashed')
    plt.axhline(y=timing[0][0][1]*par.dt,color='k',linewidth=2,linestyle='dashed')
    for k in range(par.epochs):
        for n in range(par.nn):
            plt.scatter([k]*len(z_out[n][k][b]),z_out[n][k][b],s=7)
        
#%%

for k,j in zip(z_out[0][0],range(par.epochs)):
    plt.scatter([j]*len(k),k,c='navy',s=7)