import numpy as np

class TrainerClass:
    
    def __init__(self,par,neuron,train_data,test_data):
        
        self.par = par
        self.neuron = neuron
        self.train_data = train_data[0]
        self.test_data = test_data[0]
        self.train_onset = train_data[1]
        self.test_onset = test_data[1]
        
        self.losstrainList = np.zeros((self.par.epochs,self.par.train_nb))
        self.losstestList = np.zeros((self.par.epochs,self.par.test_nb))
        self.frList = np.zeros((self.par.epochs,self.par.test_nb))
        self.vList = np.zeros((self.par.epochs,self.par.test_nb))
        self.onsetList = np.zeros((self.par.epochs,self.par.test_nb,self.par.batch))
        self.w = np.zeros((self.par.epochs,self.par.N))
        self.zList = []
        
    def _get_batch(self,data,onset):

        idx = np.arange(0,data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:self.par.batch]

        return data[idx],onset[idx]
    
    def _forward(self,x):
        
        v = np.zeros(self.par.batch)
        loss = np.zeros(self.par.batch)
        z = np.full((self.par.batch,self.par.T),False)
        
        for t in range(self.par.T):
            
            v += self.neuron.v
            loss += np.linalg.norm(x[:,:,t] - 
                            self.neuron.v*self.neuron.w)

            if self.train_FLAG == True:
                self.neuron.backward_online(x[:,:,t])  
                self.neuron.update_online() 
            
            self.neuron(x[:,:,t])
            
            z[:,t] = self.neuron.z.astype(bool)

        return z, v, loss
        
    def _do_train(self):
        
        self.train_FLAG = True
        
        loss = np.zeros(self.par.train_nb)
        for __ in range(self.par.train_nb):
            
            x,_ = self._get_batch(self.train_data,self.train_onset)
            
            self.neuron.state()
            
            _, _, loss_batch = self._forward(x)
            loss[__] = loss_batch.mean()

        return loss
    
    def _do_test(self):
        
        self.train_FLAG = False
        
        loss = np.zeros((self.par.test_nb,self.par.batch))
        v = np.zeros((self.par.test_nb,self.par.batch))
        fr = np.zeros((self.par.test_nb,self.par.batch))
        onset = np.zeros((self.par.test_nb,self.par.batch))
        z = []
        for __ in range(self.par.test_nb):
            
            x,onset_batch = self._get_batch(self.test_data)
            
            self.neuron.state()
            z_batch, v_batch, loss_batch = self._forward(x)
            
            loss[__,:] = loss_batch.mean()
            v[__,:] = v_batch.mean()
            fr[__,:] = self._get_firing_rate(z_batch).mean()
            onset[__,:] = onset_batch
            z.append(z_batch)

        return loss,v,z,fr,onset

    def train(self,log):
        
        for e in range(self.par.epochs):
            
            loss_train = self._do_train()
            loss_test, v, z, fr, onset = self._do_test()
            
            self.losstrainList[e,:] = loss_train
            self.losstestList[e,:] = loss_test
            self.vList[e,:] = v
            self.frList[e,:] = fr
            self.onsetList[e,:] = onset
            self.w[e,:] = self.neuron.w
            self.zList.append(z)
            
            self._save(log,np.mean(loss_train),
                       np.mean(loss_test))
            if e%20 == 0: 
                print('epoch {} \n loss {} \n v {} '.format(
                        e,np.mean(loss_train),np.mean(v)))
                print('weights {}'.format(self.neuron.w))
                print('fr {}'.format(fr))
                print(np.where(z[0][0,:] == True)[0]*self.par.dt)
    
    def _get_firing_rate(self,z):

        return (z.sum(axis=1)/self.par.T)*(1e3/self.par.T)

    def _save(self,log,loss_train,loss_test):
         with open(log,'a') as train:
                train.write('{}, {} \n'.format(loss_train,
                                loss_test))