import numpy as np

class TrainerClass:
    
    def __init__(self,par,network,train_data,test_data):
        """
        Args:
        par: Hyperparameters for the simulation.
        network: Network model for the simulation.
        train_data: Training data for the simulation.
        test_data: Test data for the simulation.

        Attributes:
        losstrainList: Array to store loss values for training data during each epoch.
        losstestList: Array to store loss values for test data during each epoch.
        vList: Array to store membrane voltages during each epoch.
        latencyList: Array to store latencies during each epoch.
        activityList: Array to store activity during each epoch.
        w: Array to store weights during each epoch.
        zList: List to store outputs of the network during each epoch.
        """
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
        """
        Args:
        data: The dataset to generate a batch from.
        onset: Onset times of the data (optional).

        Returns:
        data[idx]: the batch of data.
        """
        idx = np.arange(0,data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:self.par.batch]
        
        return data[idx]
    
    def _forward(self,x):
        """
        Args:
        x: Input data of shape (batch_size, n_inputs, n_neurons, time_steps).

        Returns:
        z: Output of the network as a boolean array of shape (batch_size, n_neurons, time_steps).
        v: Membrane voltages of neurons as an array of shape (batch_size, n_neurons).
        loss: Loss values for each neuron at each time step as an array of shape (batch_size, n_neurons).
        """
        
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
        """
        Returns:
        loss: Array of loss values for each neuron during training, averaged across all training batches.
        """
        
        self.train_FLAG = True
        loss = np.zeros((self.par.train_nb,self.par.nn))

        for __ in range(self.par.train_nb):    
            x = self._get_batch(self.train_data)
            self.network.state()
            _, _, loss_batch = self._forward(x)

            loss[__] = loss_batch.mean(axis=0)

        return loss
    
    def _do_test(self):
        """
        Returns:
        loss: Array of loss values for each neuron during testing, averaged across all test batches.
        v: Array of average membrane potentials for each neuron during testing.
        z: List of arrays representing the output spikes (True/False) of neurons at each time step,
           for each test batch.
        activity: Array of average duration of network activity for each test batch.
        latency: Array of average first spike times in the network for each neuron during testing.
        """
        
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
        """
        Args:
        log: Log file to save training statistics.

        Returns:
        None

        Updates the following attributes:
        losstrainList: A numpy array to store loss values for training data during each epoch.
        losstestList: A numpy array to store loss values for test data during each epoch.
        vList: A numpy array to store membrane voltages during each epoch.
        activityList: A numpy array to store duration of network activity during each epoch.
        latencyList: A numpy array to store first spike times during each epoch for each batch of test data.
        w: A numpy array to store weights during each epoch.
        zList: A list to store outputs of the network during each epoch.
        """
        
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
        """
        Args:
        z: Array representing the binary spike times (True/False) of neurons at each time step.

        Returns:
        activity: Array of duration of network activity for each batch.
        """

        activity = np.zeros(self.par.batch)

        for b in range(self.par.batch):
            if np.where(z[b,:,:])[1] != []:
                first_spk = np.where(z[b,:,:])[1].min()*self.par.dt
                last_spk = np.where(z[b,:,:])[1].max()*self.par.dt
                activity[b] = last_spk-first_spk 

        return activity
    
    def _get_latency(self,z):
        """
        Args:
        z: Array representing the binary spike times (True/False) of neurons at each time step.

        Returns:
        latency: Array of latency values (time step index) for the first spike of each neuron in each batch.
        """
        
        return np.argmin(z,axis=2)

    def _save(self,log,loss_train,loss_test):
         with open(log,'a') as train:
                train.write('{}, {} \n'.format(loss_train,
                                loss_test))