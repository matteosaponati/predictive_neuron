import numpy as np
import torch
import torch.nn as nn
import types
import matplotlib.pyplot as plt
import torch.nn.functional as F

savedir = '/gs/home/saponatim/'

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 2e-4
par.tau_m = 10.
par.v_th = 2.

par.tau_x = 2.

par.N = 2
par.N_stdp = 2
par.batch = 1
par.T = int(500/par.dt)
par.epochs = 60

'initial conditions'
w_0_pre = np.array([.001,.11])
w_0_post = np.array([.11,.001])

'optimization'
par.bound='soft'

timing = np.array([2.,6.])/par.dt

def get_sequence_stdp(par,timing):
    
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)    
    x_data[:,timing,range(par.N_stdp)]= 1
    
    'synaptic time constant'
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]

    return x_data.permute(0,2,1)


x_data  = get_sequence_stdp(par,timing)


class NeuronClass(nn.Module):
    """
    NEURON MODEL
    - get the input vector at the current timestep
    - compute the current prediction error
    - update the recursive part of the gradient and the membrane potential
    - thresholding nonlinearity and evaluation of output spike
    inputs:
        par: model parameters
    """
    
    def __init__(self,par):
        super(NeuronClass,self).__init__() 
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)              
        self.w = nn.Parameter(torch.empty(self.par.N)).to(self.par.device)
        torch.nn.init.normal_(self.w, mean=0.0, std=1/np.sqrt(self.par.N))
        
    def state(self):
        """initialization of neuron state"""
        
        self.v = torch.zeros(self.par.batch).to(self.par.device)
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.p = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.epsilon = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        self.grad = torch.zeros(self.par.batch,self.par.N).to(self.par.device)
        
    def __call__(self,x):
        """recursive dynamics step"""
        
        self.v = self.alpha*self.v + x@self.w \
                    - self.par.v_th*self.z.detach()
        self.z = torch.zeros(self.par.batch).to(self.par.device)
        self.z[self.v - self.par.v_th > 0] = 1
        
    def backward_online(self,x):
        """
        online evaluation of the gradient:
            - compute the local prediction error 
            - compute the local component of the gradient
            - update the pre-synaptic traces
        """
        
        self.epsilon =  x - self.v[:,None]*self.w[None,:]
        self.grad = -(self.v[:,None]*self.epsilon + \
                        (self.epsilon@self.w)[:,None]*self.p)
        self.p = self.alpha*self.p + x
        
    def update_online(self):
        """
        online update of parameters
        soft: apply soft lower-bound, update proportional to parameters
        hard: apply hard lower-bound, hard-coded positive parameters
        """
        if self.par.bound == 'soft':
            self.w =  nn.Parameter(self.w - 
                                   self.w*(self.par.eta*torch.mean(self.grad,dim=0)))
        if self.par.bound == 'hard':
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*torch.mean(self.grad,dim=0))
            self.w = nn.Parameter(torch.where(self.w<0,
                                       torch.zeros_like(self.w),self.w))
        else:
            self.w =  nn.Parameter(self.w - 
                                   self.par.eta*torch.mean(self.grad,dim=0))
    
    def backward_offline(self,v,x):
        """offline evaluation of the gradient"""
        
        epsilon = x - torch.einsum("bt,j->btj",v,self.w)
        filters = torch.tensor([self.alpha**(x.shape[1]-i-1) 
                                for i in range(v.shape[1])]).float().view(1, 1, -1).to(self.par.device)
        p = F.conv1d(x.permute(0,2,1),filters.expand(self.par.N,-1,-1),
                         padding=x.shape[1],groups=self.par.N)[:,:,1:x.shape[1]+1]
        ## check how to compute this
        grad = v*epsilon + epsilon@self.w*p
        return grad


neuron = NeuronClass(par)
neuron.w = nn.Parameter(torch.tensor([.001,.11])).to(par.device)

#%%
'----------------'
def forward(par,neuron,x_data):
    v,z = [], []
    for t in range(par.T):    
        v.append(neuron.v) 
        with torch.no_grad():
            neuron.backward_online(x_data[:,t])
            neuron.update_online()  
        neuron(x_data[:,t])  
        if neuron.z[0] != 0: z.append(t*par.dt)    
    return neuron, torch.stack(v,dim=1), z

def train(par,neuron,x_data):
    w1, w2 = [], []
    spk_out = []
    'training'
    for e in range(par.epochs):
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)
        'output'
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        spk_out.append(z)
        
        if e%2 == 0: print(e)
    return w1, w2, spk_out
'---------------------------------------------'

#%%

w1_pre,w2_pre,spk_prepost = train(par,neuron,x_data)


#%%
'initial condition'
w_0_pre = torch.tensor([.001,.08])
w_0_post = torch.tensor([.08,.001])
delay = np.arange(4.,40,10)


#%%

'initial conditions'
w_0_pre = torch.tensor([.001,.11])
w_0_post = torch.tensor([.11,.001])

#%%
w1,w2 = [[],[]],[[],[]]
spk = [[],[]]
for k in range(len(delay)):
    print('delay '+str(delay[k]))
    
    timing = np.array([2.,2.+ delay[k]])/par.dt
    par.T = int(delay[k]/par.dt) + 200
    x_data = get_sequence_stdp(par,timing)
    
    neuron = NeuronClass(par)
    neuron.w = nn.Parameter(w_0_pre).to(par.device)
    w1_pre,w2_pre,spk_prepost = train(par,neuron,x_data)
    
    neuron = NeuronClass(par)
    neuron.w = nn.Parameter(w_0_post).to(par.device)
    w1_post,w2_post,spk_postpre = train(par,neuron,x_data)
    
    w1[0].append(w1_pre)
    w1[1].append(w1_post)
    w2[0].append(w2_pre)
    w2[1].append(w2_post)
    spk[0].append(spk_prepost)
    spk[1].append(spk_postpre)

#%%
savedir='/gs/home/saponatim/predictive_neuron/figures/fig_stdp/'

np.save(savedir+'w1',w1)
np.save(savedir+'w2',w2)
np.save(savedir+'spk',spk)

wpre = np.load(savedir+'w_pre.npy')
wpost = np.load(savedir+'w_post.npy')
spk = np.load(savedir+'spk.npy',allow_pickle=True)
