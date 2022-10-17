"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"main_nn_selforg_example.py":
train the neural network model with learnable recurrent connections (Figure 3)

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

from predictive_neuron import models, funs_train, funs

par = types.SimpleNamespace()

'set model'
par.device = 'cpu'
par.dt = .05
par.eta = 1e-5
par.tau_m = 20.
par.v_th = 3.
par.tau_x = 2.
par.nn = 4
par.lateral = 2
par.is_rec = True

'set noise sources'
par.noise = True
par.freq_noise = True
par.freq = 10
par.jitter_noise = True
par.jitter = 1
par.batch = 1

'set input'
par.sequence = 'deterministic'
par.Dt = 2
par.n_in = 10
par.delay = 4
timing = [[] for n in range(par.nn)]
spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
for n in range(par.nn): 
    for b in range(par.batch): timing[n].append((spk_times+n*par.delay/par.dt).astype(int))

'set initialization and training algorithm'
par.init = 'fixed'
par.init_mean = 0.03
par.init_a, par.init_b = 0, .06
par.w_0rec = .000

'set training algorithm'
par.bound = 'none'
par.epochs = 400

'set noise sources'
par.T = int((par.n_in*par.delay + par.n_in*par.Dt + par.jitter + 80)/par.dt)

#%%
x = funs.get_sequence_nn_selforg_NumPy(par,timing)

#%%
plt.imshow(x[:,:,-1],aspect='auto')

#%%
       
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
        'external inputs + lateral connections'
        self.p = np.zeros((self.par.n_in+2,self.par.nn))
        self.epsilon = np.zeros((self.par.n_in+2,self.par.nn))
        self.grad = np.zeros((self.par.n_in+2,self.par.nn))  

    def __call__(self,x):
        
        'create total input'
        self.z_out = self.beta*self.z_out + self.z
        
        x_tot = np.zeros((self.par.n_in+2,self.par.nn))
        for n in range(self.par.nn): 
            if n == 0:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([0,self.z_out[n+1]])),axis=0)       
            elif n == self.par.nn-1:
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],0])),axis=0)   
            else: 
                x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],self.z_out[n+1]])),axis=0) 
                
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon = x_tot - self.w*self.v
        self.grad = self.v*self.epsilon \
                         + np.sum(self.w*self.epsilon,axis=0)*self.p
        self.p = self.alpha*self.p + x_tot
        
        'soft: apply soft lower-bound, update proportional to parameters'
        'hard: apply hard lower-bound, hard-coded positive parameters'
        
        if self.par.bound == 'soft':
            self.w = self.w + self.w*self.par.eta*self.grad
        elif self.par.bound == 'hard':
            self.w = self.w + self.par.eta*self.grad
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w + self.par.eta*self.grad
                
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.sum(x_tot*self.w,axis=0) \
                 - self.par.v_th*self.z
        self.z = np.zeros(self.par.nn)
        self.z[self.v-self.par.v_th>0] = 1

'set model'
network = NetworkClass_SelfOrg_NumPy(par)
network = funs_train.initialization_weights_nn_NumPy(par,network)
 
#%%


'training'
w,v,spk = funs_train.train_nn_NumPy(par,network,x=x)


# w,v,spk = funs_train.train_nn_NumPy(par,network,timing=timing)


 #%%

m=1
for n in range(par.nn):
    plt.eventplot(spk[n][-1],lineoffsets = m,linelengths = 1,linewidths = 3,colors = 'rebeccapurple')
    m+=1
plt.ylim(1,par.nn)
plt.xlim(0,par.T*par.dt)
plt.xlabel('time [ms]')



#%%
plt.imshow(w[-1])
plt.colorbar()


#%%

plt.plot(v[-1][-1,:])
#%%

w = np.vstack(w)
# fig = plt.figure(figsize=(7,6), dpi=300)
plt.title(r'$\vec{w}/\vec{w}_0$')
plt.ylabel(r'inputs')
plt.xlabel(r'epochs')
plt.xlim(0,par.epochs)
plt.imshow(w.T,aspect='auto',cmap='coolwarm')#,norm=MidpointNormalize(midpoint=1))
plt.colorbar()


#%%
# ## ADDING SET TIMING IN THE MAIN SCRIPT
# 'set timing'

# 'create timing'
# if par.random==True:
#     timing = [[] for n in range(par.nn)]
#     for n in range(par.nn):
#         for b in range(par.batch): 
#             spk_times = np.random.randint(0,(par.Dt/par.dt)*par.n_in,size=par.n_in)
#             timing[n].append(spk_times+n*(par.n_in*par.Dt/par.dt)+ par.delay/par.dt)
# else: 
#     timing = [[] for n in range(par.nn)]
#     spk_times = np.linspace(par.Dt,par.Dt*par.n_in,par.n_in)/par.dt
#     for n in range(par.nn):
#         for b in range(par.batch): timing[n].append(spk_times+n*par.delay/par.dt) #*(par.n_in*par.Dt/par.dt)+ 
        


'---------------------------------------------'

"""
"""

# 'fix seed'
# np.random.seed(par.seed)
    
'set model'
network = models.NetworkClass_SelfOrg_NumPy(par)
network.w = funs_train.initialization_weights_nn_NumPy(par,network)

'training'
w,v,spk = funs_train.train_nn_PyTorch(network)

