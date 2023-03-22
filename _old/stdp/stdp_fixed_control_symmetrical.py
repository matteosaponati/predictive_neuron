import numpy as np
import sys

'model parameters'
p_num = {}
p_num['dt'] = .05
p_num['tau'] = 5.
p_num['v_th'] = 2.
p_num['gamma'] = .0

'simulation parameters'
T, epochs = 200, 60
tau_x, A_x = 2, 1
delay = np.arange(0.,60,20)
#delay = [0,1,2,3,4,5]

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
        self.p, self.epsilon = np.zeros(N),  np.zeros(N)
        self.w = w_0.copy()
        self.grad_w1, self.grad_w2 = np.zeros(N), np.zeros(N)
        self.gamma = p_num['gamma']

    def num_step(self,x,p_num,idx):    
        def error(x,v,w):
            return x - w*v
        def grad_w1(epsilon,w,p):
            return np.dot(epsilon,w)*p
        def grad_w2(epsilon,v):
            return epsilon*v
        def surr_grad(v,gamma,v_th):
            return gamma*(1/(np.abs(v - v_th)+1.0)**2)   
        
        self.epsilon[idx] = error(x[idx],self.v,self.w[idx])
        
        self.grad_w1[idx], self.grad_w2[idx] = grad_w1(self.epsilon[idx],self.w[idx],self.p[idx]), grad_w2(self.epsilon[idx],self.v) 
        self.w[idx] = self.w[idx] + self.w[idx]*self.eta*(self.grad_w1[idx] + self.grad_w2[idx])
        
        self.p[idx] = ((1-self.dt/self.tau) + surr_grad(self.v,self.gamma,self.v_th))*self.p[idx]+ x[idx]
        self.v = (1-self.dt/self.tau)*self.v + np.dot(self.w,x)
        
        if self.v > self.v_th:
            self.v = self.v - self.v_th
            out = 1
        else: out = 0
        
        return out 

def train(inputs,N,epochs,T,w_0,p_num,idx):
    
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
            spk = neuron.num_step(inputs[:,t],p_num,idx)
            if spk == 1: var['v_spk'][e].append(var['time'][t])        
            'allocate variables'
            var['v'][e][t] = neuron.v
            var['w'][e][:,t] = neuron.w
        'reinitialize'
        neuron = neuron_module(N,p_num,neuron.w)
    
    return var

#%%
    

'pre-post'
w_0_pre = np.array([.001,.11])
w_0_post = np.array([.11,.001])

w_pre,w_post = [],[]
spk_pre, spk_post = [], []

for k in range(len(delay)):
    'set pairing'
    timing = np.array([2.,2.+ delay[k]])
    print('timing {}'.format(timing))
    inputs = sequence(len(timing),timing,A_x,tau_x,T,p_num)
    'numerical solution'
    p_num['eta'] = 3e-4
    var_pre = train(inputs,inputs.shape[0],epochs,T,w_0_pre,p_num,0)
    var_post= train(inputs,inputs.shape[0],epochs,T,w_0_post,p_num,1)
    'get weights'
    w_pre.append(var_pre['w'][-1][0,-1])
    w_post.append(var_post['w'][-1][1,-1])
    
    spk_pre.append(var_pre['v_spk'])
    spk_post.append(var_post['v_spk'])

    del var_pre,var_post    

#%%
    
#%%
plt.plot(np.array(w_pre)/w_0_pre[0])
plt.plot(np.array(w_post)/w_0_post[1])    

#%%
savedir='/gs/departmentN4/matteo_data/predictive_neuron/stdp/classical_stdp/'
np.save(savedir+'w_pre_fixed_control',w_pre)
np.save(savedir+'w_post_fixed_control',w_post)
