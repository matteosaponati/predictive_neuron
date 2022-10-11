"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"stdp_froemke2006_frequency.py"
'Froemke et al (2006) Contribution of inidividual spikes in burst-induced 
long-term synaptic modification. Journal of Neuroscience'

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

#from predictive_neuron import models, funs, funs_train

'set model'
par = types.SimpleNamespace()
par.device = 'cpu'
par.dt = .05
par.eta = 8e-5
par.tau_m = 40.
par.v_th = 3.
par.tau_x = 2.
par.bound = 'soft'

'set inputs'
par.N = 2
par.T = int(600/par.dt)
par.epochs = 30

'initial conditions'
w_0 = np.array([.14,.018])

"""
we reproduce the experimental protocol by increasing the number of inputs from
the second pre-synaptic neurons
input:
    
"""
n_spk = 5
dt_burst, dt = int(10/par.dt), int(5/par.dt)

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
w_post,w_tot = [],[]
spk_out = []
for k in np.arange(1,n_spk+1):

    'set inputs'
    timing = [(np.arange(0,10*k,10)/par.dt).astype(int),dt]
    x_data = get_sequence_stdp(par,timing)
    'numerical solutions'
    neuron = NeuronClass_NumPy(par)
    neuron.w = w_0.copy()
    w1,w2,v,spk = train(par,neuron,x_data)
    'get weights'
    w_tot.append(w2)
    w_post.append(w2[-1])
    spk_out.append(spk)


#%%
    
for k in range(len(w_post)):
    plt.plot(w_tot[k]/w_0[1],label='{}'.format(k))
plt.legend()
#%%
    
'plots'
#savedir = '/Users/saponatim/Desktop/predictive_neuron/paper_review/fig_stdp/'
#fig = plt.figure(figsize=(6,6), dpi=300)
plt.axhline(y=1, color='black',linestyle='dashed',linewidth=1.5)
plt.plot(np.arange(1,n_spk+1),np.array(w_post)/w_0[1],color='rebeccapurple',linewidth=2)
'add experimental data'
x = [1,2,3,4,5]
y, y_e = [.7,.8,.9,1.02,1.2],[.1,.1,.1,.05,.05]
plt.scatter(x,y,color='k',s=20)
plt.errorbar(x,y,yerr = y_e,color='k',linestyle='None')
#fig.tight_layout(rect=[0, 0.01, 1, 0.96])
#plt.ylabel(r'$w/w_0$')
#plt.xlabel(r'# spikes')
#plt.xticks(np.arange(1,n_spk+1),np.arange(1,n_spk+1))
#plt.ylim(.5,1.5)
#plt.savefig(savedir+'/stdp_froemke2006.pdf', format='pdf', dpi=300)
#plt.savefig(savedir+'/stdp_froemke2006.png', format='png', dpi=300)
#plt.close('all')
#%%
"RMS error"
y = [.7,.8,.9,1.02,1.2]
error = np.sqrt(np.sum((np.array(w_post)/w_0[1] - np.array(y))**2)/len(y))