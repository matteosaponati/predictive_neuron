"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"selforganization_replay.py"
compute how many neurons are needed for replay

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import numpy as np
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

from predictive_neuron import funs_train

par = types.SimpleNamespace()

'set model'
par.device = 'cpu'
par.dt = .05
par.eta = 1e-5
par.tau_m = 15.
par.v_th = 3.5
par.tau_x = 2.
par.nn = 8
par.lateral = 2
par.is_rec = True

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 2
par.batch = 1

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 26
par.delay = 4
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append((spk_times+n*par.delay/par.dt).astype(int))

'set total lenght of the simulation'
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

'---------------------------------------------'

"""
quantification of the number of neurons that needs to be activate such that
the network can recall the whole sequence. We show that the number of neurons 
required for the recall decreases consistently across epochs. 

During each epoch we use *par.nn* pre-synaptic inputs 
"""

def get_sequence_nn_selforg_NumPy(par,timing):
    
    'loop on neurons in the network'
    x_data  = []
    for n in range(par.nn):

        'add background firing'         
        if par.freq_noise == True:
            prob = (np.random.randint(0,par.freq,par.n_in)*par.dt)/1000
            x = np.zeros((par.n_in,par.T))
            for nin in range(par.n_in): x[nin,:][np.random.rand(par.T)<prob[nin]] = 1        
        else:
            x = np.zeros((par.n_in,par.T))
        
        'span across neurons in the network'
        if n in par.subseq:
        
            'create sequence + jitter'
            if par.jitter_noise == True:
                timing_err = np.array(timing[n]) \
                              +  (np.random.randint(-par.jitter,par.jitter,par.n_in)/par.dt).astype(int)
                x[range(par.n_in),timing_err] = 1
            else: x[range(par.n_in),timing[n]] = 1
        
        'synaptic time constant'
        for nin in range(par.n_in):
            x[nin,:] = np.convolve(x[nin,:],
                          np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]   
            
        'add to total input'
        x_data.append(x)

    return np.stack(x_data,axis=2)

"""
Next, we use the set of synaptic inputs which the network learned when we showed
the whole input sequence. Consequently, we define a novel NetworkClass where the 
forward pass does not contain the update step for the synaptic weights. For each
training, we assign to the network the set of synaptic weights learned 
at the respective epoch.
"""

class NetworkClass_SelfOrg_NumPy():
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
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        self.w = np.zeros((self.par.n_in+self.par.lateral,self.par.nn))
        
    def state(self):
        """initialization of network state"""

        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)

    def __call__(self,x):
        
        'create total input'
        x_tot = np.zeros((self.par.n_in+2,self.par.nn))
        self.z_out = self.beta*self.z_out + self.z
        for n in range(self.par.nn): 
            if n == 0:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([0,self.z_out[n+1]])),axis=0)       
            elif n == self.par.nn-1:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],0])),axis=0)   
            else: 
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],self.z_out[n+1]])),axis=0) 
                
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.sum(x_tot*self.w,axis=0) \
                 - self.par.v_th*self.z
        self.z = np.zeros(self.par.nn)
        self.z[self.v-self.par.v_th>0] = 1


"""
Next, we define a novel NetworkClass where the forward pass does not contain 
the update step for the synaptic weights. For each training epoch, we assign 
to the network the set of synaptic weights learned at the respective epoch.
We then show to the network only the first input in the pre-synaptic sequence 
and we check how many neurons in the network were active. We gradually increase
the number of inputs in the pre-synaptic sequence until the network reaches
a full recall. 
"""

'load synaptic weights'
w = np.load(par.loaddir+'w_recall.npy')

'count of the number of neurons required for full recall, across epochs'
n_required = np.zeros(w.shape[0])

count = 0
for e in range(w.shape[0]):
    
    print('epoch '+str(e))
    
    'run across neurons in the network'
    for subseq in range(1,par.nn+1):
        
        print('# neurons: '+str(subseq))

        'span on the possible subsequences'
        par.subseq = [x for x in range(subseq)]
        
        'define sequence recall'
        timing = [[] for n in range(par.nn)]
        spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
        for n in range(par.nn): 
            for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt)
    
        'create input'
        x_subseq = get_sequence_nn_selforg_NumPy(par,timing)
    
        'set model'
        network = NetworkClass_SelfOrg_NumPy(par)
        network = w[e,:].copy()
                
        'training'
        w,v,spk = funs_train.train_nn_NumPy(par,network,x=x_subseq)
        
        '''
        span the spiking activity of every neuron in the network
        count if the neuron has been active during the simulaion
        '''
        count = 0
        for n in range(par.nn):
            if spk[n] != []: count+=1
        
        '''
        check if every neuron in the network was active during the simulation
        if true, the current value of *subseq* represent the amount of input
        needed for the network to recall the whole sequence.
        if false, the number of input presented to the network is not sufficient
        to trigger the recall of the whole sequence.
        '''
        print('total neurons triggered: '+str(count))
        if count == par.nn:
            n_required[e] = subseq
            break
        else:
            continue


'---------------------------------------'
'plot'

fig = plt.figure(figsize=(5,6), dpi=300)
plt.plot(n_required,linewidth=2,color='purple')
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel('epochs')
plt.ylabel(r'# neurons for replay')
plt.ylim(0,9)
plt.savefig(par.savedir+'n_needed.png',format='png', dpi=300)
plt.savefig(par.savedir+'n_needed.pdf',format='pdf', dpi=300)
plt.close('all')           


#
#
#'---------------------------------------'
#import torch.nn.functional as F
#def get_subseqs(par,timing):            
#    x_data  = []
#    
#    for n in range(par.nn):
#
#        x = torch.zeros(par.batch,par.T,par.n_in).to(par.device)
#        if n in par.subseq:
#         x[b,timing[n][b],range(par.n_in)] = 1
#        
#        'filtering'
#        filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
#                               for i in range(par.T)]).view(1,1,-1).float().to(par.device) 
#        x = F.conv1d(x.permute(0,2,1),filter.expand(par.n_in,-1,-1),
#                          padding=par.T,groups=par.n_in)[:,:,1:par.T+1]
#            
#        'add to total input'
#        x_data.append(x.permute(0,2,1))
#
#    return torch.stack(x_data,dim=3)
#
#def forward(par,network,x_data):
#    z = [[[] for b in range(par.batch)] for n in range(par.nn)]
#    v = []
#    for t in range(par.T):     
#        'append voltage state'
#        v.append(network.v.clone().detach().numpy())
#        'forward pass'
#        network(x_data[:,t]) 
#        'append output spikes'
#        for n in range(par.nn):
#            for b in range(par.batch):
#                if network.z[b,n] != 0: z[n][b].append(t*par.dt)          
#        
#    return network, np.stack(v,axis=1), z
#'---------------------------------------'
#
#'---------------------------------------'
#'create parameter structure'
#par = types.SimpleNamespace()
#'architecture'
#par.n_in = 26
#par.nn = 8
#par.batch = 1
#par.lateral = 2
#par.device = 'cpu'
#par.dtype = torch.float
#'model parameters'
#par.dt = .05
#par.tau_m = 15.
#par.v_th = 3.5
#par.tau_x = 2.
#par.is_rec = True
#par.online = True
#'input'
#par.Dt=2
#par.delay = 4
#par.jitter_noise = True
#par.jitter = 2
#par.fr_noise = True
#par.freq = .01
#par.w_0 = .02
#par.T = int((par.n_in*par.delay+(par.n_in*par.Dt)+80)/par.dt)
#
#par.loaddir = ''
#par.savedir = '/Users/saponatim/Desktop/'
#
#'---------------------------------------'
#'get results'
#z_out = np.load(par.loaddir+'spk_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
#                                    par.n_in,par.nn,par.delay,par.Dt,
#                                    par.tau_m,par.v_th,par.w_0),allow_pickle=True).tolist()
#w = np.load(par.loaddir+'w_nin_{}_nn_{}_delay_{}_Dt_{}_tau_{}_vth_{}_w0_{}.npy'.format(
#                                    par.n_in,par.nn,par.delay,par.Dt,par.tau_m,par.v_th,par.w_0))
#
#'---------------------------------------'
