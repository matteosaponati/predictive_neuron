"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"nn_selforg_example.py":
train the neural network model with learnable recurrent connections (Figure 3)

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.stats as stats
import types
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)
import matplotlib.colors as colors
class MidpointNormalize(colors.Normalize):
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

par = types.SimpleNamespace()

'set model'
par.device = 'cpu'
par.dt = .05
par.eta = 1e-5
par.tau_m = 20.
par.v_th = 3.
par.tau_x = 2.
par.nn = 8
par.lateral = 2
par.is_rec = True
par.bound = 'none'

'set noise sources'
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.batch = 1

'set input'
par.Dt = 2
par.n_in = 2
par.delay = 4
par.T = int((par.n_in*par.delay+(par.n_in*par.Dt)+80)/par.dt)
'define input sequence'
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)

'set initialization and training algorithm'
par.init = 'random'
par.init_mean = 0.01
par.init_a, par.init_b = 0, .08
par.w_0rec = .003

'set training algorithm'
par.seed = 1992
par.
par.bound = 'soft'
par.epochs = 1000

#%%
def get_sequence_nn_selforg(par):
    
    'create timing'
    if par.random==True:
        timing = [[] for n in range(par.nn)]
        for n in range(par.nn):
            for b in range(par.batch): 
                spk_times = np.random.randint(0,(par.Dt/par.dt)*par.n_in,size=par.n_in)
                timing[n].append(spk_times+n*(par.n_in*par.Dt/par.dt)+ par.delay/par.dt)
    else: 
        timing = [[] for n in range(par.nn)]
        spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
        for n in range(par.nn):
            for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt) #*(par.n_in*par.Dt/par.dt)+ 
            
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.freq_noise == True:
            prob = par.freq*par.dt
            mask = torch.rand(par.batch,par.T,par.n_in).to(par.device)
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            x[mask<prob] = 1        
        else:
            x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
            
        'create sequence + jitter' 
        for b in range(par.batch):
            if par.jitter_noise == True:
                timing_err = np.array(timing[n][b]) + np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt
                x[b,timing_err,range(par.n_in)] = 1
            else: x[b,timing[n][b],range(par.n_in)] = 1
        
        'filtering'
        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
            
        'add to total input'
        x_data.append(x.permute(0,2,1))

    return torch.stack(x_data,dim=3)


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
                             + torch.sum(self.w*self.epsilon[b,:],dim=0)*self.p[b,:])
        self.p = self.alpha*self.p + x_tot
        
    def update_online(self):
        self.w =  nn.Parameter(self.w - 
                               self.par.eta*torch.mean(self.grad,dim=0))

#%%


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

def train(par):
    
    'create input data'
    x_data = get_sequence_nn_selforg(par)
    
    'set model'
    network = NetworkClass_SelfOrg(par)
    
    'initialization'
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.n_in+par.lateral,par.nn)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=.1/np.sqrt(par.par.n_in+par.lateral),
                                    a=par.init_a,b=par.init_b) 
        network.w[par.n_in:,] = par.w_0rec    
    if par.init == 'fixed':
        w = par.w_0*torch.ones(par.n_in+par.lateral,par.nn)
        w[par.n_in:,] = par.w_0rec
        network.w = nn.Parameter(w).to(par.device)
    
    'allocate outputs'
    w = np.zeros((par.epochs,par.n_in+par.lateral,par.nn))
    z_out = [[] for n in range(par.nn)]
    v_out = []
    
    'training'
    for e in range(par.epochs):
        if e%50 == 0: print(e)  
            
        network.state()
        network, v, z = forward(par,network,x_data)
        v_out.append(v)
        
        w[e,:,:] = network.w.detach().numpy()
        for n in range(par.nn):
            z_out[n].append(z[n])

    return w, z_out, v_out

#%%
    


'---------------------------------------------'

"""
there are three sources of noise for each epoch:
    1. jitter of the spike times (random jitter between -par.jitter and +par.jitter)
    2. random background firing following an homogenenous Poisson process with rate
    distributione between 0 and par.freq 
    3. another subset of N_dist pre-synaptic neurons that fire randomly according
    to an homogenenous Poisson process with randomly distribuited rates between
    0 and par.freq
"""

'fix seed'
np.random.seed(par.seed)
    
'set model'
neuron = NetworkClass_SelfOrg(par)
if par.init == 'fixed': 
    neuron.w = par.init_mean*np.ones(par.N)
if par.init == 'random':
    neuron.w = stats.truncnorm((par.init_a-par.init_mean)/(1/np.sqrt(par.N)), 
                          (par.init_b-par.init_mean)/(1/np.sqrt(par.N)), 
                          loc=par.init_mean, scale=1/np.sqrt(par.N)).rvs(par.N)

'training'
w_out = np.zeros((par.epochs,par.N))
spk_out = []
v_out = []
onset = []
loss_tot = []

for e in range(par.epochs):
    
    'get input data'
    onset.append(np.random.randint(0,par.T/2))
    x_data = funs.get_sequence_NumPy(par,timing,onset[e])
    
    'numerical solution'
    neuron.state()
    neuron, v, z , loss = funs_train.forward_NumPy(par,neuron,x_data)
    
    'output'
    w_out[e,:] = neuron.w.copy()
    spk_out.append(z)
    v_out.append(v)
    loss_tot.append(np.sum(loss))
    if e%50 == 0: print(e)

