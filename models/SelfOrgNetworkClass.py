import numpy as np

class NetworkClassNumPy():
    
    def __init__(self,par):
        
        self.par = par  
        self.alpha = (1-self.par.dt/self.par.tau_m)  
        self.beta = (1-self.par.dt/self.par.tau_x)              
        
        if self.par.network_type == 'nearest':
            self.w = np.zeros((self.par.n_in+self.par.lateral,self.par.nn))

        elif self.par.network_type == 'all':
            self.w = np.zeros((self.par.n_in+self.par.nn,self.par.nn))

    def state(self):
        
        self.v = np.zeros(self.par.nn)
        self.z = np.zeros(self.par.nn)
        self.z_out = np.zeros(self.par.nn)
        
        if self.par.network_type == 'nearest':
            self.p = np.zeros((self.par.n_in+2,self.par.nn))
            self.epsilon = np.zeros((self.par.n_in+2,self.par.nn))
            self.grad = np.zeros((self.par.n_in+2,self.par.nn))  

        elif self.par.network_type == 'all':
            self.p = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
            self.epsilon = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
            self.grad = np.zeros((self.par.n_in+self.par.nn,self.par.nn))

    def __call__(self,x):
        
        self.z_out = self.beta*self.z_out + self.z

        if self.par.network_type == 'nearest':

            x_tot = np.zeros((self.par.n_in+2,self.par.nn))
        
            for n in range(self.par.nn): 
                if n == 0:
                    x_tot[:,n] = np.concatenate((x[:,n],np.array([0,self.z_out[n+1]])),axis=0)       
                elif n == self.par.nn-1:
                    x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],0])),axis=0)   
                else: 
                    x_tot[:,n] = np.concatenate((x[:,n],np.array([self.z_out[n-1],self.z_out[n+1]])),axis=0) 

        elif self.par.network_type == 'all':

            'create total input'
            x_tot = np.zeros((self.par.n_in+self.par.nn,self.par.nn))
            self.z_out = self.beta*self.z_out + self.z
            for n in range(self.par.nn): 
                x_tot[:,n] = np.concatenate((x[:,n],np.append(np.delete(self.z_out,
                                             n,axis=0),[0],axis=0)),axis=0)  
                
        self.epsilon = x_tot - self.w*self.v
        self.grad = self.v*self.epsilon \
                         + np.sum(self.w*self.epsilon,axis=0)*self.p
        self.p = self.alpha*self.p + x_tot
        
        if self.par.bound == 'soft':
            self.w = self.w + self.w*self.par.eta*self.grad
        elif self.par.bound == 'hard':
            self.w = self.w + self.par.eta*self.grad
            self.w = np.where(self.w<0,np.zeros_like(self.w),self.w)
        else: self.w = self.w + self.par.eta*self.grad
                
        self.v = self.alpha*self.v + np.sum(x_tot*self.w,axis=0) \
                 - self.par.v_th*self.z
        self.z = np.zeros(self.par.nn)
        self.z[self.v-self.par.v_th>0] = 1