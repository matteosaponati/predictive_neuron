import numpy as np

class NetworkClassNumPy():
    
    def __init__(self,par):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)       

    def initialize(self):

        self.w = np.zeros((self.par.N,self.par.nn))

        if self.par.network_type == 'nearest':
            self.w[:self.par.n_in,:] = self.par.init_mean*np.ones((self.par.n_in,self.par.nn))
            self.w[self.par.n_in:,:] = self.par.init_rec*np.ones((2,self.par.nn))

        if self.par.network_type == 'all':
            self.w[:self.par.n_in,:] = self.par.init_mean*np.ones((self.par.n_in,self.par.nn))
            self.w[self.par.n_in:,:] = self.par.init_rec*np.ones((self.par.nn,self.par.nn))
        
        if self.par.network_type == 'random':
            self.w[:self.par.N_in,:] = np.random.uniform(0.,self.par.init_mean,(self.par.N_in,self.par.nn))
            self.w[self.par.N_in:,:] = np.random.uniform(0.,self.par.init_rec,(self.par.nn,self.par.nn))

    def state(self):

        self.v = np.zeros((self.par.batch,self.par.nn))
        self.z = np.zeros((self.par.batch,self.par.nn))
        self.z_out = np.zeros((self.par.batch,self.par.nn))
        
        self.p = np.zeros((self.par.batch,self.par.N,self.par.nn))
        self.epsilon = np.zeros((self.par.batch,self.par.N,self.par.nn))
        self.grad = np.zeros((self.par.batch,self.par.N,self.par.nn))  

    def __call__(self,x):

        self._get_inputs(x)
        
        self.v = self.alpha*self.v + np.einsum('ijk,jk->ik',self.x,self.w) \
                    - self.par.v_th*self.z
        
        self.z = np.zeros((self.par.batch,self.par.nn))
        self.z[self.v - self.par.v_th > 0] = 1

    def backward_online(self,x):

        self._get_inputs(x)
        
        self.epsilon =  self.x - np.einsum('ij,kj->ikj',self.v,self.w)
        self.heterosyn = np.einsum('ikj,kj->ij',self.epsilon,self.w)
        self.grad = - np.einsum('ij,ikj->ikj',self.v,self.epsilon) \
                    - np.einsum('ij,ikj->ikj', self.heterosyn,self.p)
        self.p = self.alpha*self.p + self.x

    def update_online(self):

        if self.par.bound == 'soft':
            self.w = self.w - self.w*self.par.eta*self.grad.mean(axis=0)
        elif self.par.bound == 'hard':
            self.w = self.w - self.par.eta*self.grad.mean(axis=0)
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w - self.par.eta*self.grad.mean(axis=0)


    def _get_inputs(self, x):
        self.z_out = self.beta * self.z_out + self.z
        self.x = np.zeros((self.par.batch,self.par.N,self.par.nn))
    
        if self.par.network_type == 'nearest':

            for n in range(self.par.nn): 
                if n == 0:
                    self.x[:,:,n] = np.hstack((x[:,:,n],
                                                 np.zeros((self.par.batch,1)),
                                                 self.z_out[:,n+1][:,None]))
                elif n == self.par.nn - 1:
                    self.x[:,:,n] = np.hstack((x[:,:,n],
                                                 self.z_out[:,n-1][:,None],
                                                 np.zeros((self.par.batch,1))))
                else:
                    self.x[:,:,n] = np.hstack((x[:, :, n],
                                                 self.z_out[:,n-1][:,None],
                                                 self.z_out[:,n+1][:,None]))
        else:
            for n in range(self.par.nn):
                temp_z_out = np.delete(self.z_out, n, axis=1)
                self.x[:,:,n] = np.hstack((x[:,:,n], temp_z_out, np.zeros((self.par.batch,1))))