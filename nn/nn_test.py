import numpy as np
import torch
import types
import torch.nn as nn
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

from predictive_neuron import models_nn, funs

'----------------'
def forward(par,network,x_data):
    
    v,z = [], []
    
    for t in range(par.T):            

        v.append(network.v)              

        network(x_data[:,t]) 
        
        z.append(network.z_out.detach())
        
    return network, torch.stack(v,dim=1), torch.stack(z,dim=1)
'----------------'


#%%










savedir = ''

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.nn = 2
par.T = 600
par.batch = 1
par.epochs = 100
par.device = 'cpu'
par.dtype = torch.float

'model parameters'
par.dt = .05
par.eta = 1e-6
par.tau_m = 20.
par.v_th = 2.
par.tau_x = 2.

par.is_rec = False

'set inputs'
timing = np.array([2.,6.])/par.dt
par.Dt = 4
timing = np.cumsum(np.random.randint(0,par.Dt,par.N))/par.dt
x_data  = []
for n in range(par.nn):
    x,_ = funs.get_sequence(par,timing + n*(4/par.dt))
    print(timing + n*(4/par.dt))
    x_data.append(x)
x_data = torch.stack(x_data,dim=3)
    
# x_data,density = funs.get_sequence(par,timing)

'set model'
network = models_nn.NetworkClass(par)
par.w_0 = .03
par.w_0rec = .0
network.w = nn.Parameter(par.w_0*torch.ones(par.N,par.nn)).to(par.device)
if par.is_rec == True: 
    w_rec=  par.w_0rec*np.ones((par.nn,par.nn))
    w_rec = np.where(np.eye(par.nn)>0,np.zeros_like(w_rec),w_rec)
    network.wrec = nn.Parameter(torch.as_tensor(w_rec,dtype=par.dtype).to(par.device))

#%%

'setup optimization'

loss_fn = nn.MSELoss(reduction='sum')

optimizerList = []
for n in range(par.nn):
    optimizer = torch.optim.Adam(network.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    optimizerList.append(optimizer)

#%%

network.state()
network, v, z = forward(par,network,x_data)

#%%
'feedforward network'
# optimizer = torch.optim.Adam(network.parameters(),
#                               lr=par.eta,betas=(.9,.999))

optimizer = torch.optim.SGD(network.parameters(),lr=par.eta)

w = np.zeros((par.epochs,par.N,par.nn))
E = [[] for n in range(par.nn)]

for e in range(par.epochs):
    
    # for n in range(par.nn): 
    #     optimizerList[n].zero_grad()
    
    optimizer.zero_grad()
        
    network.state()
    network, v, z = forward(par,network,x_data)
    
    x_hat = torch.einsum("btn,jn->btjn",v,network.w)
    
    lossList = []
    for n in range(par.nn):  
        loss = loss_fn(x_hat[:,:,:,n],x_data)
        lossList.append(loss)
        
    for n in range(par.nn): 
        lossList[n].backward(retain_graph = True)
        E[n].append(lossList[n].item())
    
        
    # for n in range(par.nn): 
    #     optimizerList[n].step()
    
    optimizer.step()
        
    w[e,:,:] = network.w.detach().numpy()
    
    if e%50 == 0:
        for n in range(par.nn):
            
            print('loss {}: {}'.format(n,lossList[n].item()))

#%%
# optimizer = torch.optim.Adam(network.parameters(),
#                               lr=par.eta,betas=(.9,.999))

optimizer = torch.optim.SGD(network.parameters(),lr=par.eta)

'recurrent network'
w = np.zeros((par.epochs,par.N,par.nn))
wrec = np.zeros((par.epochs,par.nn,par.nn))

v_out, spk_out = [], []
E = [[] for n in range(par.nn)]

for e in range(par.epochs):
    
    # for n in range(par.nn): 
    #     optimizerList[n].zero_grad()
        
    optimizer.zero_grad()
        
    network.state()
    network, v, z = forward(par,network,x_data)
    
    wtot = torch.vstack([network.w,network.wrec])
    x_hat = torch.einsum("btn,jn->btjn",v,wtot)
    
    lossList = []
    for n in range(par.nn):  
        xtot = torch.cat([x_data[:,:,:,n],z.detach()],dim=2)
        loss = loss_fn(x_hat[:,:,:,n],xtot)
        lossList.append(loss)
        
    for n in range(par.nn): 
        lossList[n].backward(retain_graph = True)
        E[n].append(lossList[n].item())
        
        
    # for n in range(par.nn): 
    #     optimizerList[n].step()
    
    network.wrec.grad = torch.where(torch.eye(par.nn)>0,
                                    torch.zeros_like(network.wrec.grad),
                                    network.wrec.grad)
    optimizer.step()
        
    w[e,:,:] = network.w.detach().numpy()
    wrec[e,:,:] = network.wrec.detach().numpy()
    
    v_out.append(v.detach().numpy())
    spk_out.append(z.detach().numpy())
    
    if e%50 == 0:
        for n in range(par.nn):
            
            print('loss {}: {}'.format(n,lossList[n].item()))


#%%
chosen = 1

plt.plot(wrec[:600,0,chosen],'k')
plt.plot(wrec[:600,1,chosen],'navy')

plt.plot(w[:600,0,chosen],'r')
plt.plot(w[:600,1,chosen],'g')

#%%
chosen = 3
for n in range(par.nn):
    plt.plot(wrec[:,n,chosen])


for n in range(par.N):
    plt.plot(w[:,n,chosen])


#%%

for n in range(par.nn):
    plt.plot(v_out[450][0,:,n])


#%%
lossList = []
for n in range(par.nn):  
    loss = loss_fn(x_hat[:,:,:,n],xtot)
    lossList.append(loss)
    
for n in range(par.nn): 
    lossList[n].backward(retain_graph = True)
for n in range(par.nn): 
    optimizerList[n].step()
    

#%%
    loss = nn.MSELoss(reduction='sum')
    
    'initialization'
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=.1/np.sqrt(par.N),
                                    a=par.init_a,b=par.init_b)
    if par.init == 'fixed':
        network.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
        
    'optimizer'
    optimizer = torch.optim.Adam(network.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    
    'allocate outputs'
    loss_out = []
    w = np.zeros((par.epochs,par.N))
    v_out, spk_out = [], []
    
    for e in range(par.epochs):
        
        network.state()
        network, v, z = forward(par,networkn,x_data)
        
        'evaluate loss'
        for k in range(par.nn):
            
            x_hat = torch.einsum("bt,j->btj",v,neuron.w)
            E = .5*loss(x_hat,x_data)
            
        optimizer.zero_grad()    
        
        
        
        E = .5*loss(x_hat,x_data)
        
        E.backward()
        optimizer.step()
        
        'output'
        loss_out.append(E.item())
        w[e,:] = neuron.w.detach().numpy()
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))




























#%%

'----------------'
def train(par,x_data):
    
    'fix seed'
    torch.manual_seed(par.seed)
    torch.cuda.manual_seed(par.seed)
    np.random.seed(par.seed)
    
    'set model'
    network = models_nn.NetworkClass(par)
    loss = nn.MSELoss(reduction='sum')
    
    'initialization'
    if par.init == 'trunc_gauss':
        network.w = nn.Parameter(torch.empty(par.N)).to(par.device)
        torch.nn.init.trunc_normal_(network.w, mean=par.init_mean, std=.1/np.sqrt(par.N),
                                    a=par.init_a,b=par.init_b)
    if par.init == 'fixed':
        network.w = nn.Parameter(par.w_0*torch.ones(par.N)).to(par.device)
        
    'optimizer'
    optimizer = torch.optim.Adam(network.parameters(),
                                  lr=par.eta,betas=(.9,.999))
    
    'allocate outputs'
    loss_out = []
    w = np.zeros((par.epochs,par.N))
    v_out, spk_out = [], []
    
    for e in range(par.epochs):
        
        network.state()
        network, v, z = forward(par,networkn,x_data)
        
        'evaluate loss'
        for k in range(par.nn):
            
            x_hat = torch.einsum("bt,j->btj",v,neuron.w)
            E = .5*loss(x_hat,x_data)
            
        optimizer.zero_grad()    
        
        
        
        E = .5*loss(x_hat,x_data)
        
        E.backward()
        optimizer.step()
        
        'output'
        loss_out.append(E.item())
        w[e,:] = neuron.w.detach().numpy()
        v_out.append(v.detach().numpy())
        spk_out.append(z)
        
        if e%50 == 0: 
            print('epoch {} loss {}'.format(e,E.item()))
    
    return loss_out, w, v_out, spk_out



savedir = '/gs/home/saponatim/'

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.nn = 2
par.T = 300
par.batch = 1
par.epochs = 100
par.device = 'cpu'

'model parameters'
par.dt = .05
par.eta = 3e-5
par.tau_m = 10.
par.v_th = 2.5
par.tau_x = 2.

'set inputs'
timing = np.array([2.,6.])/par.dt
x_data = funs.get_sequence(par,timing)