

'architecture'
par.N = 2
par.nn = 2
par.T = 600
par.batch = 1
par.epochs = 1000
par.device = 'cpu'
par.dtype = torch.float


    'training algorithm'
    parser.add_argument('--optimizer',type=str, 
                        choices=['online','SGD','Adam'],default='Adam',
                        help='choice of optimizer')
    parser.add_argument('--bound',type=str,
                        choices=['None','hard','soft'],default='None',
                        help='set hard lower bound for parameters')
    parser.add_argument('--init',type=str, 
                        choices=['random','fixed'],default='random',
                        help='type of weights initialization')
    
    parser.add_argument('--init_mean',type=float, default=0.1)
    parser.add_argument('--init_a',type=float, default=0.)
    parser.add_argument('--init_b',type=float, default=.2)

    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1992)
    parser.add_argument('--batch', type=int, default=10)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--rep', type=int, default=1)
    
    'set input sequence'
    parser.add_argument('--n_in', type=int, default=100)
    parser.add_argument('--Dt', type=int, default=2) 
    
    'set noise sources'
    parser.add_argument('--noise', type=bool, default=True)
    parser.add_argument('--freq_noise', type=bool, default=True)
    parser.add_argument('--freq', type=float, default=10) 
    parser.add_argument('--jitter_noise', type=bool, default=True) 
    parser.add_argument('--jitter', type=float, default=2)
    parser.add_argument('--onset', type=bool, default=False)
    
    'network model'
    parser.add_argument('--is_rec', type=bool, default=True)
    parser.add_argument('--nn', type=int, default=10)
    parser.add_argument('--w0_rec', type=float, default=-.05) 
    parser.add_argument('--dt', type=float, default= .05) 
    parser.add_argument('--tau_m', type=float, default= 10.) 
    parser.add_argument('--v_th', type=float, default= 2.)
    parser.add_argument('--eta',type=float, default=1e-3)
    parser.add_argument('--tau_x', type=float, default= 2.)
    parser.add_argument('--dtype', type=str, default=torch.float) 
    
    parser.add_argument('--upload_data', type=bool, default=False)  
    parser.add_argument('--load_dir', type=str, default='') 
    parser.add_argument('--save_dir', type=str, default='')


savedir = ''

par = types.SimpleNamespace()

'architecture'
par.N = 2
par.nn = 2
par.T = 600
par.batch = 1
par.epochs = 1000
par.device = 'cpu'
par.dtype = torch.float

'model parameters'
par.dt = .05
par.eta = 1e-6
par.tau_m = 20.
par.v_th = 2.
par.tau_x = 2.

par.is_rec = True

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