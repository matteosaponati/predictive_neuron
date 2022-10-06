"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_nevian2006.py"
'Nevian et al (2006) Spine Ca2+ signaling in spike-timing-dependent plasticity
Journal of Neuroscience'

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

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 1.8e-4
par.tau_m = 10.
par.v_th = 2.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N  = 2
par.T = int(300/par.dt)
par.epochs = 60

"""
we reproduce the experimental protocol by increasing the frequency of post bursts
inputs:
    1. n_spk: total number of post spikes in the bursts
    2. dt_burst, dt: delay between post spikes, delay between pre and first post
"""
n_spikes = 3
dt_burst, dt = (np.array([10.,20.,50.])/par.dt).astype(int), int(10./par.dt)

'set initial conditions'
w_0_pre = np.array([.01,.08,])
w_0_post = np.array([.08,.01])

#%%

'create input pattern'
def get_sequence_stdp(par,timing):    
    x_data = np.zeros((par.N,par.T))
    for n in range(par.N):
        x_data[n,timing[n]]= 1
        x_data[n,:] = np.convolve(x_data[n,:],
                      np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]        
    return x_data
  
class NeuronClass_NumPy():
    """
    NEURON MODEL (Numpy version - online update)
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
        self.w = np.zeros(self.par.N)
        
    def state(self):
        """initialization of neuron state"""
        
        self.v = 0
        self.z = 0
        self.p = np.zeros(self.par.N)
        self.epsilon = np.zeros(self.par.N)
        self.grad = np.zeros(self.par.N)
    
    def __call__(self,x):
        
        'compute prediction error (eq 4) and update parameters (eq 3)'
        self.epsilon =  x - self.w*self.v
        self.grad = self.v*self.epsilon + np.dot(self.epsilon,self.w)*self.p
        if self.par.bound == 'soft':
            self.w = self.w + self.w*self.par.eta*self.grad
        elif self.par.bound == 'hard':
            self.w = self.w + self.par.eta*self.grad
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w + self.par.eta*self.grad
        
        'update eligibility traces'
        self.p = self.alpha*self.p + x
        
        'update membrane voltage (eq 1)'
        self.v = self.alpha*self.v + np.dot(x,self.w) 
        if self.v-self.par.v_th>0: 
            self.z = 1
            self.v = self.v - self.par.v_th
        else: self.z = 0

#%%
        
'----------------'
def forward(par,neuron,x_data):
    
    v,z = [], []
    for t in range(par.T):    
        v.append(neuron.v) 
        neuron(x_data[:,t])  
        
        if neuron.z != 0: z.append(t*par.dt)    
    return neuron, v, z


def train(par,neuron,x_data):
    w1, w2 = [], []
    spk_out = []
    v_out = []
    'training'
    for e in range(par.epochs):
        
        neuron.state()
        neuron, v, z = forward(par,neuron,x_data)
        
        'output'
        w1.append(neuron.w[0].item())
        w2.append(neuron.w[1].item())
        spk_out.append(z)
        v_out.append(v)
        if e%10 == 0: print(e)
        
    return w1, w2, v_out, spk_out
'---------------------------------------------'

#%%

'training (pre-post protocol)'
w_pre,w_post = [],[]
for j in dt_burst:
    
    print('solving {} dt'.format(j))
    
    'pre-post protocol'
    timing = [np.array(0),np.arange(dt,j*n_spikes + j,j)] 
    x_data = get_sequence_stdp(par,timing)
    neuron = NeuronClass_NumPy(par)
    neuron.w = w_0_pre
    w1,w2,v,spk = train(par,neuron,x_data)
    w_pre.append(w1[-1])
    
    'post-pre protocol'
    timing = [np.arange(0,j*n_spikes + j,j),np.array(j*n_spikes+ dt)]     
    x_data = get_sequence_stdp(par,timing)
    neuron = NeuronClass_NumPy(par)
    neuron.w = w_0_post
    w1,w2,v,spk = train(par,neuron,x_data)
    w_pre.append(w2[-1])

#%%

savedir = '/Users/saponatim/Desktop/'
fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(1e3/dt_burst[::-1],np.array(w_pre)[::-1]/w_0_pre[0],color='royalblue',linewidth=2,label='pre-post')
#plt.plot(1e3/dt_burst[::-1],np.array(w_post)[::-1]/w_0_post[1],color='rebeccapurple',linewidth=2,label='post-pre')
'add experimental data'
x = [20,50,100]
y_pre, y_pre_e = [1.1,2,2.25],[.3,.3,.6]
plt.scatter(x,y_pre,color='k',s=20)
plt.errorbar(x,y_pre,yerr = y_pre_e,color='k',linestyle='None')
y_post, y_post_e = [.74,.74,.55],[.2,.1,.15]
plt.scatter(x,y_post,color='k',s=20)
plt.errorbar(x,y_post,yerr = y_post_e,color='k',linestyle='None')

fig.tight_layout(rect=[0, 0.01, 1, 0.96])
plt.xlabel(r'frequency [Hz]')
plt.ylabel(r'$w/w_0$')
plt.savefig(savedir+'/burst_effect_stdp.png', format='png', dpi=300)
plt.savefig(savedir+'/burst_effect_stdp.pdf', format='pdf', dpi=300)
plt.close('all')