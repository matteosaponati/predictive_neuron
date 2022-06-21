import numpy as np
import sys
dir = ''
maindir = dir+'fig1/'
sys.path.append(dir), sys.path.append(maindir)

'-----------------------------------------------------------------------------'
'CLASSICAL STDP WINDOW'

'model parameters'
p_num = {}
p_num['dt'] = .05
p_num['eta'] = 1e-4
p_num['v_th'] = 2.
p_num['gamma'] = .0

'simulation parameters'
T, epochs = 200, 40
tau_x, A_x = 2, 1

'initial conditions'
w_0_pre = np.array([.005,.1])
w_0_post = np.array([.11,.01])
factor = np.arange(0,1.2,.2)

#%%

'preallocation of relevant variables'
def preallocation(N,T,epochs,p_num):
    var = {}
    var['time'] = np.arange(0,T,p_num['dt'])
    var['v'] = [np.zeros_like(var['time']) for k in range(epochs)]
    var['w'] = [np.zeros((N,len(var['time']))) for k in range(epochs)]
    var['loss'] = np.zeros(epochs)
    var['v_spk'] = [[] for k in range(epochs)]
    return var

'create input pattern'
def sequence(N,timing,size,tau,T,p_num):    
    n_steps = int(T/p_num['dt'])
    inputs = np.zeros((N,n_steps))
    for k in range(N):
        inputs[k,np.array(timing[k]/p_num['dt']).astype('int')]= size
        inputs[k,:] = np.convolve(inputs[k,:],np.exp(-np.arange(0,T,p_num['dt'])/tau))[:n_steps]        
    return inputs

class neuron_module:
    
    def __init__(self,N,p_num,w_0):
        self.eta, self.tau, self.v_th, self.dt = p_num['eta'], p_num['tau'], p_num['v_th'], p_num['dt']
        self.N = N
        self.v = 0
        self.vfactor = 0
        self.p, self.epsilon = np.zeros(N),  np.zeros(N)
        self.w = w_0.copy()
        self.grad_w1, self.grad_w2 = np.zeros(N), np.zeros(N)
        self.gamma = p_num['gamma']

    def num_step(self,x,p_num,factor):    
        def error(x,v,w,):
            return x - w*v
        def grad_w1(epsilon,w,p):
            return np.dot(epsilon,w)*p
        def grad_w2(epsilon,v):
            return epsilon*v
        def surr_grad(v,gamma,v_th):
            return gamma*(1/(np.abs(v - v_th)+1.0)**2)   
        
        'compute prediction error (eq. 10)'
        self.epsilon = error(x,self.vfactor,self.w)
        'compute gradient and update weight vector (eq 14 and eq 16)'
        self.grad_w1, self.grad_w2  = grad_w1(self.epsilon,self.w,self.p), grad_w2(self.epsilon,self.vfactor) 
        self.w = self.w + self.w*self.eta*(self.grad_w1 + self.grad_w2)
#        self.w = self.w + self.eta*(self.grad_w1 + self.grad_w2)
        'update recursive part of the gradient (eq 8)'
        self.p = ((1-self.dt/self.tau) + surr_grad(self.v,self.gamma,self.v_th))*self.p + x
        'update membrane voltage (eq 2)'
        self.v = (1-self.dt/self.tau)*self.v + np.dot(self.w,x)
        self.vfactor = (1-self.dt/self.tau)*self.vfactor + np.dot(self.w,x)
        
        if self.v > self.v_th:
            self.v = self.v - self.v_th
            self.vfactor = self.vfactor - factor
            out = 1
        else: out = 0
        
        return out 

def train(inputs,N,epochs,T,w_0,p_num,factor):
    
    'objective function'
    def loss(x,w,v):
        return np.linalg.norm(x-v*w)
    'preallocation of variables'
    var = preallocation(N,T,epochs,p_num)
    neuron = neuron_module(N,p_num,w_0)
    
    for e in range(epochs):
        'numerical solution'
        for t in range(int(T/p_num['dt'])):
            var['loss'][e] += loss(inputs[:,t],neuron.v,neuron.w)
            spk = neuron.num_step(inputs[:,t],p_num,factor)
            if spk == 1: var['v_spk'][e].append(var['time'][t])        
            'allocate variables'
            var['v'][e][t] = neuron.v
            var['w'][e][:,t] = neuron.w
        'reinitialize'
        neuron = neuron_module(N,p_num,neuron.w)
    
    return var

#%%
    
p_num['tau'] = 20.

print(p_num['tau'])
w_pre,w_post = [],[]
spk_pre, spk_post = [], []

timing = np.array([2.,8.])
inputs = sequence(len(timing),timing,A_x,tau_x,T,p_num)

var_pre = train(inputs,inputs.shape[0],epochs,T,w_0_pre,p_num,0*p_num['v_th'])
w_pre=[]
for k in range(epochs):
    w_pre.append(var_pre['w'][k][0,-1])
plt.plot(np.array(w_pre)/w_0_pre[0])
#var_post= train(inputs,inputs.shape[0],epochs,T,w_0_post,p_num,0)

#%%
w_pre = []
for k in np.arange(0,1.1,.1):
    print(k)
    var_pre = train(inputs,inputs.shape[0],epochs,T,w_0_pre,p_num,k*p_num['v_th'])
    w_pre.append(var_pre['w'][-1][0,-1])
    
plt.plot(np.array(w_pre)[::-1]/w_0_pre[0])

#%%
import matplotlib.pyplot as plt

w_pre,w_post = [], []
for k in range(epochs):
    w_pre.append(var_pre['w'][k][0,-1])
#    w_post.append(var_post['w'][k][1,-1])
    
#%%
    

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
plt.rc('axes', axisbelow=True)

savedir = '/Users/saponatim/Desktop/predictive_neuron/paper_review/fig_stdp/'

fig = plt.figure(figsize=(6,6), dpi=300)
plt.plot(np.arange(0,1.1,.1)[::1],np.array(w_pre)[::-1]/w_0_pre[0],color='purple',linewidth=2)
fig.tight_layout(rect=[0, 0.01, 1, 0.97])
plt.xlabel(r'scaling factor')
plt.ylabel(r'$w/w_0$')
plt.axhline(y=1,color='k',linestyle='dashed')
plt.savefig(savedir+'stdp_membrane_potential.png',format='png', dpi=300)
plt.savefig(savedir+'stdp_membrane_potential.pdf',format='pdf', dpi=300)
plt.close('all')


#%%

for k in range(len(factor)):
    'set pairing'
    
    'numerical solution'
    var_pre = train(inputs,inputs.shape[0],epochs,T,w_0_pre,p_num,factor[k])
    var_post= train(inputs,inputs.shape[0],epochs,T,w_0_post,p_num,factor[k])
    'get weights'
    w_pre.append(var_pre['w'][-1][0,-1])
    w_post.append(var_post['w'][-1][1,-1])
    
    spk_pre.append(var_pre['v_spk'])
    spk_post.append(var_post['v_spk'])

    del var_pre,var_post    