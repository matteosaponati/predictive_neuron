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
        v.append(network.v)
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
        
    return network, torch.stack(v,dim=1), z
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
        
        if self.par.selforg == True:
            self.w = nn.Parameter(torch.empty((self.par.n_in+self.par.nn-1,self.par.nn)).to(self.par.device))
            torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in))
        else: 
            self.w = nn.Parameter(torch.empty((self.par.n_in,self.par.nn)).to(self.par.device))
            torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.n_in))
        
    def state(self):
        """initialization of network state"""
        
        self.v = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        self.z_out = torch.zeros(self.par.batch,self.par.nn).to(self.par.device)
        
        self.p = torch.zeros(self.par.batch,self.par.n_in+self.par.nn-1,self.par.nn).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.n_in+self.par.nn-1,self.par.nn).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.n_in+self.par.nn-1,self.par.nn).to(self.par.device)  

    def __call__(self,x):
        
        'create total input'
        x_tot = torch.zeros(self.par.batch,self.par.n_in+self.par.nn-1,self.par.nn).to(self.par.device)
        self.z_out = self.beta*self.z_out + self.z.detach()
        for n in range(self.par.nn):
            x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,:n],
                                                   self.z_out.detach()[:,n+1:]],dim=1)],dim=1)
        
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
        x_tot = torch.zeros(self.par.batch,self.par.n_in+self.par.nn-1,self.par.nn).to(self.par.device)
        for n in range(self.par.nn):
            x_tot[:,:,n] = torch.cat([x[:,:,n],
                                        torch.cat([self.z_out.detach()[:,:n],
                                                   self.z_out.detach()[:,n+1:]],dim=1)],dim=1)
        
        x_hat = torch.zeros(self.par.batch,self.par.n_in+self.par.nn-1,self.par.nn)
        for b in range(self.par.batch):
            x_hat[b,:] = self.w*self.v[b,:]
            self.epsilon[b,:] = x_tot[b,:] - x_hat[b,:]
            self.grad[b,:] = -(self.v[b,:]*self.epsilon[b,:] \
                             + torch.sum(self.w*self.epsilon[0,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x_tot
        
    def update_online(self):
        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))
    
        self.w = nn.Parameter(torch.where(self.w>self.par.w_max,
                                          self.par.w_max*torch.ones_like(self.w),
                                          self.w))
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
par.n_in = 2
par.nn = 2
par.T = 1000
par.batch = 1
par.epochs = 1000
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
par.selforg = True

par.w_max = .05
#%%
'setup inputs'
timing = [[] for n in range(par.nn)]
spk_times = []
for b in range(par.batch):
    spk_times.append(np.random.randint(0,par.T-200,size=par.n_in))
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b])

x_data = get_pattern_fixed_nn(par,timing)

#%%

'setup inputs'
delay = 4/par.dt
timing = [[] for n in range(par.nn)]
spk_times = []
for b in range(par.batch):
    spk_times.append(np.cumsum(np.random.randint(0,4,par.n_in))/par.dt)
for n in range(par.nn):
    for b in range(par.batch): timing[n].append(spk_times[b]+n*delay)

x_data = get_pattern_fixed_nn(par,timing)

#%%

plt.imshow(x_data[0,:,:,0].T,aspect='auto')

#%%
    
# x_data,density = funs.get_sequence(par,timing)

'set model'
network = NetworkClass_SelfOrg(par)
par.w_0 = .001
par.w_0rec = .0
network.w = nn.Parameter(par.w_0*torch.ones(par.n_in+par.nn-1,par.nn)).to(par.device)

#%%

def forward_check(par,network,x_data):
    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
    v = []
    p = []
    epsilon = []
    
    for t in range(par.T):     
        'append voltage state'
        v.append(network.v.detach())
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
        
        p.append(network.p.detach())
        epsilon.append(network.epsilon.detach())
        
    return network, torch.stack(v,dim=1), z, torch.stack(p,dim=1), torch.stack(epsilon,dim=1)

network.state()
network,v,z,p,epsilon = forward_check(par,network,x_data)

# plt.plot(v[0,:,1].detach().numpy())    
plt.plot(p[0,:,-1,1].detach().numpy())    

#%%

'setup optimization'

loss_fn = nn.MSELoss(reduction='sum')

optimizerList = []
for n in range(par.nn):
    optimizer = torch.optim.Adam(network.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    optimizerList.append(optimizer)

#%%
w = np.zeros((par.epochs,par.n_in+par.nn-1,par.nn))
E = [[] for n in range(par.nn)]
z_out = [[] for n in range(par.nn)]
v_out = []

optimizer = torch.optim.SGD(network.parameters(),lr=par.eta)

for e in range(par.epochs):
        
    optimizer.zero_grad()
        
    network.state()
    network, v, z = forward(par,network,x_data)
        
    v_out.append(v.detach().numpy())
#    x_hat = torch.einsum("btn,jn->btjn",v,network.w)    
#    lossList = []
#    for n in range(par.nn):  
#        loss = loss_fn(x_hat[:,:,:,n],x_data)
#        lossList.append(loss)        
#    for n in range(par.nn): 
#        lossList[n].backward(retain_graph = True)
#        E[n].append(lossList[n].item())    
#    optimizer.step()
    
    w[e,:,:] = network.w.detach().numpy()
    for n in range(par.nn):
        z_out[n].append(z[n])
    
    if e%50 == 0: print(e)
    
#%%
    
for n in range(par.nn):
    plt.subplot(par.nn,1,n+1)
    for k in range(par.n_in):
        plt.plot(w[:,k,n],linewidth=2)#,label='neuron {}'.format(k+1))
    for k in range(par.n_in,par.n_in+par.nn-1):
        plt.plot(w[:,k,n],linewidth=2,linestyle='dashed')#,label='neuron {}'.format(k+1))
plt.legend()

#%%

plt.plot(v_out[-1][0,:,0])
    
#%%

plt.subplot(1,2,1)
for k,j in zip(z_out[0],range(par.epochs)):
    plt.scatter([j]*len(k[0]),k[0],edgecolor='royalblue',facecolor='none',s=7)
for k in spk_times[0]:
    plt.axhline(y=k*par.dt,color='k')
plt.xlabel(r'epochs')
plt.ylabel('spk times [ms]')
plt.subplot(1,2,2)
for k,j in zip(z_out[1],range(par.epochs)):
    plt.scatter([j]*len(k[0]),k[0],edgecolor='royalblue',facecolor='none',s=7)
for k in spk_times[0]:
    plt.axhline(y=(k+delay)*par.dt,color='k')
plt.xlabel(r'epochs')
plt.ylabel('spk times [ms]')
    