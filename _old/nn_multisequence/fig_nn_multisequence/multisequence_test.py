import numpy as np
import os
import torch
import types
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

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

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device
'------------------'

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

def get_pattern_nn(par):
    
    x_data  = []
    for n in range(par.nn):
        x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
        prob = par.freq*par.dt
        x[par.mask<prob] = 1
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
        x_data.append(x.permute(0,2,1))
    
    return torch.stack(x_data,dim=3)

def get_pattern_fixed_nn(par,timing):
    
    x_data  = []
    for n in range(par.nn):
        x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
        for b in range(par.batch):
             x[b,timing[n][b],range(par.n_in)] = 1
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
        x_data.append(x.permute(0,2,1))

    return torch.stack(x_data,dim=3)

#%%

par = types.SimpleNamespace()

'architecture'
par.n_in = 100
par.nn = 10
par.T = 2000
par.batch = 2
par.Dt = 4
par.epochs = 100
par.device = 'cpu'
par.dtype = torch.float

'model parameters'
par.eta = 1e-5
par.dt = .05
par.tau_m = 30
par.v_th = 3.
par.tau_x = 2.

par.is_rec = True
par.online = True
par.selforg = True

par.w_0rec = -.05

#%%

'setup inputs'
spk_times = []
for b in range(par.batch):
    times = np.linspace(0,par.T-600,par.n_in,dtype=int)
    np.random.shuffle(times)
    spk_times.append(times.tolist())
    # spk_times.append(np.random.randint(0,par.T-500,size=par.n_in).tolist())
timing = [[] for n in range(par.nn)]
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b])
x_data = get_pattern_fixed_nn(par,timing)

#%%

plt.imshow(x_data[1,:,:,1].T,aspect='auto')
plt.yticks(np.arange(par.n_in)[::2],np.arange(1,par.n_in+1)[::2])
plt.xticks(np.arange(par.T)[::200],np.linspace(0,par.T*par.dt,par.T)[::200].astype(int))
plt.xlabel('time [ms]')
plt.ylabel('pre-syn inputs')

#%%

'set model'
network = NetworkClass(par)
par.w_0 = .05
network.w = nn.Parameter(torch.FloatTensor(par.n_in,par.nn).uniform_(0.,par.w_0))
if par.is_rec == True: 
    w_rec = par.w_0rec*np.ones((par.nn,par.nn))
    w_rec = np.where(np.eye(par.nn)>0,np.zeros_like(w_rec),w_rec)
    network.wrec = torch.as_tensor(w_rec,dtype=par.dtype).to(par.device)
    
    
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
                
    w[e,:,:] = network.w.detach().numpy()
    if par.is_rec == True: wrec[e,:,:] = network.wrec.detach().numpy()
    for n in range(par.nn):
        z_out[n].append(z[n])
    
    if e%50 == 0: print(e)
    
#%%

selectivity = np.zeros((par.nn,par.batch,par.epochs))

for b in range(par.batch):
    for n in range(par.nn):
        for e in range(par.epochs):
            if z_out[n][e][b] != []: selectivity[n,b,e] = 1
            
            
#%%
n = -1
plt.imshow(selectivity[n,:],aspect='auto')
        

#%%
# fig = plt.figure(figsize=(20,5))

n = 5
for b in range(par.batch):    
    plt.subplot(1,par.batch,b+1)
    
    spk = []
    for k in range(par.epochs): spk.append(z_out[n][k][b])
    for k,j in zip(spk,range(par.epochs)):
        plt.scatter([j]*len(k),k,edgecolor='royalblue',facecolor='none',s=7)
    for k in timing[n][b]:
        plt.axhline(y=(k)*par.dt,color='k')
        plt.ylim(0,par.T*par.dt)
        
#%%

for n in range(par.nn):
    plt.subplot(par.nn,1,n+1)
    for k in range(par.n_in):
        plt.plot(w[:,k,n],linewidth=2)#,label='neuron {}'.format(k+1))
plt.legend()

#%%


'set metrics'
selective_n = np.zeros((len(par.nn),len(par.inh),par.rep))
responsive_n = np.zeros((len(par.nn),len(par.inh),par.rep))
unresponsive_n = np.zeros((len(par.nn),len(par.inh),par.rep))

"""
GET SELECTIVITY
obtain a tensor that, for each repetition, computes the total amount 
of selective neurons.
"""

'running over total network size'
for nnn in range(len(par.nn)):
    'running over inhibition strength'
    for inh in range(len(par.inh)):
        
        selectivity_tot = []
        
        for rep in range(par.rep):
            
            'get selectivity on each batch'
            selectivity = np.zeros((par.batch,par.nn[nnn]))
            for b in range(par.batch):
                for n in range(par.nn[nnn]):
                    if z_out[n][-1][b] != []: selectivity[b,n] = 1
            
            'compute percentage of selective neurons'
            for n in range(par.nn[nn]):
                if selectivity[:,n].sum(axis=0) < par.batch and selectivity[:,n].sum(axis=0) > 0: selective_n[nn,inh,rep] += 1/par.nn[nn]
                if selectivity[:,n].sum(axis=0) == par.batch: responsive_n[nn,inh,rep] += 1/par.nn[nn]    
#                    if selectivity[:,n].sum(axis=0) == 0: unresponsive_n[nn,inh,rep] += 1/par.nn[nn] 