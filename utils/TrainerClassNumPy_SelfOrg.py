import numpy as np

class TrainerClass:
    
    def __init__(self,par,network,train_data,test_data):
        
        self.par = par
        self.network = network
        self.train_data = train_data
        self.test_data = test_data
        
        self.losstrainList = np.zeros((self.par.epochs,self.par.train_nb,self.par.nn))
        self.losstestList = np.zeros((self.par.epochs,self.par.test_nb,self.par.nn))
        self.vList = np.zeros((self.par.epochs,self.par.test_nb,self.par.nn))
        self.latencyList = np.zeros((self.par.epochs,self.par.test_nb,self.par.nn))
        self.activityList = np.zeros((self.par.epochs,self.par.test_nb))
        self.w = np.zeros((self.par.epochs,self.par.N,self.par.nn))
        self.zList = []
        
    def _get_batch(self,data):

        idx = np.arange(0,data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:self.par.batch]
        
        return data[idx]
    
    def _forward(self,x):
        
        v = np.zeros((self.par.batch,self.par.nn))
        loss = np.zeros((self.par.batch,self.par.nn))
        z = np.full((self.par.batch,self.par.nn,self.par.T),False)
        
        for t in range(self.par.T):
            
            v += self.network.v

            if self.train_FLAG == True:
                self.network.backward_online(x[:,:,:,t])  
                self.network.update_online() 
            
            self.network(x[:,:,:,t])
            
            z[:,:,t] = self.network.z.astype(bool)
            loss += np.linalg.norm(self.network.x - 
                           np.einsum('ij,kj->ikj',self.network.v,self.network.w))

        return z, v, loss
        
    def _do_train(self):
        
        self.train_FLAG = True
        
        loss = np.zeros((self.par.train_nb,self.par.nn))
        for __ in range(self.par.train_nb):
            
            x = self._get_batch(self.train_data)
            
            self.network.state()
            
            _, _, loss_batch = self._forward(x)
            loss[__] = loss_batch.mean(axis=0)

        return loss
    
    def _do_test(self):
        
        self.train_FLAG = False
        
        loss = np.zeros((self.par.test_nb,self.par.nn))
        v = np.zeros((self.par.test_nb,self.par.nn))
        activity = np.zeros(self.par.test_nb)
        latency = np.zeros((self.par.test_nb,self.par.nn))
        z = []
        for __ in range(self.par.test_nb):
            
            x = self._get_batch(self.test_data)
            
            self.network.state()
            z_batch, v_batch, loss_batch = self._forward(x)
            
            loss[__,:] = loss_batch.mean(axis=0)
            v[__,:] = v_batch.mean(axis=0)
            activity[__] = self._get_activity(z_batch).mean()
            latency[__,:] = self._get_latency(z_batch).mean(axis=0)

            z.append(z_batch)

        return loss,v,z,activity,latency

    def train(self,log):
        
        for e in range(self.par.epochs):
            
            loss_train = self._do_train()
            loss_test, v, z, activity, latency = self._do_test()
            
            self.losstrainList[e,:] = loss_train
            self.losstestList[e,:] = loss_test
            self.vList[e,:] = v
            self.activityList[e,:] = activity
            self.latencyList[e,:] = latency
            self.w[e,:] = self.network.w
            self.zList.append(z)
            
            self._save(log,np.mean(loss_train),
                       np.mean(loss_test))
            if e%20 == 0: 
                print('epoch {} \n loss {} '.format(
                        e,np.mean(loss_train)))
    
    def _get_activity(self,z):

        activity = np.zeros(self.par.batch)

        for b in range(self.par.batch):

            if np.where(z[b,:,:])[1] != []:

                first_spk = np.where(z[b,:,:])[1].max()*self.par.dt
                last_spk = np.where(z[b,:,:])[1].max()*self.par.dt
                
                activity[b] = last_spk-first_spk 

        return activity
    
    def _get_latency(self,z):
        return np.argmin(z,axis=2)

    def _save(self,log,loss_train,loss_test):
         with open(log,'a') as train:
                train.write('{}, {} \n'.format(loss_train,
                                loss_test))