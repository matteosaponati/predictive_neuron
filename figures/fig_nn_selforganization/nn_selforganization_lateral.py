import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)
import torch.nn.functional as F

#%%

"""
import forward
import network class
import input functions
"""

'----------------'
def forward(par,network,x_data):
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    
    for t in range(par.T):     
        'append voltage state'
        v.append(network.v.clone().detach().numpy())
        'update weights online'
        if par.online == True: 
            with torch.no_grad():
                network.backward_online(x_data[:,t])
                network.update_online()  
        'forward pass'
        network(x_data[:,t]) 
        'append output spikes'
        for n in range(par.nn):
            for b in range(par.batch):
                if network.z[b,n] != 0: z[n][b].append(t*par.dt)  
        
    return network, np.stack(v,axis=1), z
'----------------'

class NetworkClass_SelfOrg(nn.Module):
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
        super(NetworkClass_SelfOrg,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        
        self.w = nn.Parameter(torch.empty((self.par.n_in+self.par.lateral,self.par.nn)).to(self.par.device))
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in+self.par.lateral))
        
    def state(self):
        """initialization of network state"""
        
        self.v = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        
        'external inputs + lateral connections'
        self.p = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)  

    def __call__(self,x):
        
        'create total input'
        x_tot = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        self.z_out = self.beta*self.z_out + self.z.detach()
        
        for n in range(self.par.nn):
            if n == 0:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([torch.zeros(self.par.batch,1),
                                                   self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1)   
            if n == self.par.nn-1:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                  torch.zeros(self.par.batch,1)],dim=1)],dim=1)   
            else:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                            torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                       self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1)        
        'update membrane voltages'
        for b in range(self.par.batch):
            self.v[b,:] = self.alpha*self.v[b,:] + torch.sum(x_tot[b,:]*self.w,dim=0) \
                     - self.par.v_th*self.z[b,:].detach()
        
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
        
        'create total input'
        x_tot = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn).to(self.par.device)
        for n in range(self.par.nn):
            if n == 0:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([torch.zeros(self.par.batch,1),
                                                   self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1)   
            if n == self.par.nn-1:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                  torch.zeros(self.par.batch,1)],dim=1)],dim=1)   
            else:
                x_tot[:,:,n] = torch.cat([x[:,:,n],
                                            torch.cat([self.z_out.detach()[:,n-1].unsqueeze(1),
                                                       self.z_out.detach()[:,n+1].unsqueeze(1)],dim=1)],dim=1) 
        
        x_hat = torch.zeros(self.par.batch,self.par.n_in+2,self.par.nn)
        for b in range(self.par.batch):
            x_hat[b,:] = self.w*self.v[b,:]
            self.epsilon[b,:] = x_tot[b,:] - x_hat[b,:]
            self.grad[b,:] = -(self.v[b,:]*self.epsilon[b,:] \
                             + torch.sum(self.w*self.epsilon[0,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x_tot
        
    def update_online(self):
        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))
        
        
#%%
def get_sequence_nn_selforg(par,random=False):
    
    if random==True:
        timing = [[] for n in range(par.nn)]
        for n in range(par.nn):
            for b in range(par.batch): 
                spk_times = np.random.randint(0,(par.Dt/par.dt)*par.n_in,size=par.n_in)
                timing[n].append(spk_times+n*(par.delay/par.dt))
    else: 
        timing = [[] for n in range(par.nn)]
        spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
        for n in range(par.nn):
            for b in range(par.batch): timing[n].append(spk_times+n*(par.delay/par.dt))
            
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
par.n_in = 2
par.nn = 6
par.batch = 1
par.epochs = 200
par.device = 'cpu'
par.dtype = torch.float

'model parameters'
par.eta = 1e-5
par.dt = .05
par.tau_m = 20
par.v_th = 3.
par.tau_x = 2.

par.is_rec = True
par.online = True
par.random = False

#%%

'NEW INPUT'
'setup inputs'
par.delay = 4
par.Dt = 2
par.T = int((par.nn*par.delay*par.dt + par.Dt+70)/par.dt)

'create input data'
x_data = get_sequence_nn_selforg(par,random=par.random)

plt.imshow(x_data[0,:,:,3].T,aspect='auto')

#%%

'OLD INPUT'
'setup inputs'
par.delay = 4/par.dt
par.Dt = 2
par.T = int((par.nn*par.delay*par.dt + par.Dt+70)/par.dt)
timing = [[] for n in range(par.nn)]

def get_sequence_nn_selforg(par,random=False):
    
    if random==True:
        timing = [[] for n in range(par.nn)]
        for n in range(par.nn):
            for b in range(par.batch): 
                spk_times = np.random.randint(0,(par.Dt/par.dt)*par.n_in,size=par.n_in)
                timing[n].append(spk_times+n*par.delay)
    else: 
        timing = [[] for n in range(par.nn)]
        spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
        for n in range(par.nn):
            for b in range(par.batch): timing[n].append(spk_times+n*par.delay)
            
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


x_data = get_sequence_nn_selforg(par,random=False)

#%%

plt.imshow(x_data[0,:,:,0].T,aspect='auto')

#%%
'setup inputs'

'FIRST VERSION INPUT'

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


par.delay = 4/par.dt
par.Dt = 2
par.T = int((par.nn*par.delay*par.dt + par.Dt+70)/par.dt)
timing = [[] for n in range(par.nn)]
spk_times = []
times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
# times = np.random.randint(0,par.T-200,size=par.n_in)
for b in range(par.batch):
    spk_times.append(times)
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b]+n*par.delay)

x_data = get_pattern_fixed_nn(par,timing)

plt.imshow(x_data[0,:,:,0].T,aspect='auto')
#%%
    
# x_data,density = funs.get_sequence(par,timing)

par.lateral = 2
'set model'
network = NetworkClass_SelfOrg(par)
par.w_0 = .08
par.w_0rec = .0003
w = par.w_0*torch.ones(par.n_in+par.lateral,par.nn)
w[par.n_in:,] = 0.0003
network.w = nn.Parameter(w).to(par.device)

#%%

w = np.zeros((par.epochs,par.n_in+par.lateral,par.nn))
E = [[] for n in range(par.nn)]
z_out = [[] for n in range(par.nn)]
v_out = []

for e in range(par.epochs):
        
        
    network.state()
    network, v, z = forward(par,network,x_data)
        
    v_out.append(v)
    
    w[e,:,:] = network.w.detach().numpy()
    for n in range(par.nn):
        z_out[n].append(z[n])
    
    if e%50 == 0: print(e)  
    
    #%%
    
# fig = plt.figure(figsize=(10,30), dpi=300)

for n in range(par.nn):
    plt.subplot(par.nn,1,n+1)
    for k in range(par.n_in):
        plt.plot(w[:,k,n],linewidth=2)#,label='neuron {}'.format(k+1))
    for k in range(par.n_in,par.n_in+2):
        plt.plot(w[:,k,n],linewidth=2,linestyle='dashed')#,label='neuron {}'.format(k+1))

#%%

fig = plt.figure(figsize=(10,30), dpi=300)

for n in range(w.shape[-1]):
    plt.subplot(w.shape[-1],1,n+1)
    for k in range(w.shape[1]-2):
        plt.plot(w[:,k,n],linewidth=2)#,label='neuron {}'.format(k+1))
    for k in range(w.shape[1]-2,w.shape[1]):
        plt.plot(w[:,k,n],linewidth=2,linestyle='dashed')


#%%

fig = plt.figure(figsize=(20,5))
for n in range(par.nn):
    plt.subplot(1,par.nn,n+1)
    for k,j in zip(z_out[n],range(par.epochs)):
        plt.scatter([j]*len(k[0]),k[0],edgecolor='royalblue',facecolor='none',s=7)
    for k in timing[n][0]:
        plt.axhline(y=(k)*par.dt,color='k')
        plt.ylim(0,par.T*par.dt)
        
    #%%
    
n = -1
plt.plot(v_out[-1][0,:,n])
for k in timing[n][0]:
    plt.axvline(x=(k),color='k')
# plt.xlim(timing[n][0][0],timing[n][0][0])