import numpy as np
import torch
import torch.nn as nn

loss = nn.MSELoss()

class TrainerClass:
    
    def __init__(self,par,neuron,optimizer,train_data,test_data,
                 train_onset=None,test_onset=None):
        """
        Args:
        par: Hyperparameters for the simulation.
        neuron: Neuron model for the simulation.
        train_data: Training data for the simulation.
        test_data: Test data for the simulation.
        train_onset: Onset time for training data (optional).
        test_onset: Onset time for test data (optional).

        Attributes:
        losstrainList: Array to store loss values for training data during each epoch.
        losstestList: Array to store loss values for test data during each epoch.
        frList: Array to store firing rates during each epoch.
        vList: Array to store membrane voltages during each epoch.
        onsetList: Array to store onset times during each epoch for each batch of test data.
        w: Array to store weights during each epoch.
        zList: List to store outputs of the network during each epoch.
        """
        self.par = par
        self.neuron = neuron
        self.optimizer = optimizer
        self.train_data = train_data
        self.test_data = test_data
        self.train_onset = train_onset
        self.test_onset = test_onset
        self.losstrainList = np.zeros((self.par.epochs,self.par.train_nb))
        self.losstestList = np.zeros((self.par.epochs,self.par.test_nb))
        self.frList = np.zeros((self.par.epochs,self.par.test_nb))
        self.vList = np.zeros((self.par.epochs,self.par.test_nb))
        self.onsetList = np.zeros((self.par.epochs,self.par.test_nb,self.par.batch))
        self.w = np.zeros((self.par.epochs,self.par.N))
        self.zList = []
        
    def _get_batch(self,data,onset=None):
        """
        Args:
        data: The dataset to generate a batch from.
        onset: Onset times of the data (optional).

        Returns:
        If onset is provided, returns a tuple containing the batch of data and the corresponding onset times.
        If onset is not provided, returns the batch of data.
        """
        idx = np.arange(0,data.shape[0])
        np.random.shuffle(idx)
        idx = idx[:self.par.batch]
        if onset is not None: return data[idx],onset[idx]
        else: return data[idx]
    
    def _forward(self,x):
        """
        Args:
        x: A torch.Tensor of shape (batch_size, input_size, T) representing input data.

        Returns:
        z: A numpy array of shape (batch_size, T) representing the output spike times
        v: A numpy array of shape (batch_size,) representing the cumulative sum of neuron voltages.
        v_torch: A torch.Tensor of shape (batch_size, T) representing the neuron voltages at each time step.
        """
        
        v = torch.zeros(self.par.batch)
        v_torch = []
        z = np.full((self.par.batch,self.par.T),False)

        for t in range(self.par.T):
            
            v += self.neuron.v
            v_torch.append(self.neuron.v)
            
            self.neuron(x[:,:,t])
            
            z[:,t] = self.neuron.z.detach().numpy().astype(bool)

        return z, v, torch.stack(v_torch,dim=1)
        
    def _do_train(self):
        """
        Returns:
        loss: A numpy array of shape (train_nb,) representing the loss for each training iteration.
        """
        
        self.train_FLAG = True
        loss = np.zeros(self.par.train_nb)

        for __ in range(self.par.train_nb):    
            if self.train_onset is not None:
                x,_ = self._get_batch(self.train_data,
                                      self.train_onset)
            else:
                x = self._get_batch(self.train_data)

            self.neuron.state()
            _, _, v_torch = self._forward(x)

            x_hat = torch.einsum("bt,j->bjt",v_torch,self.neuron.w)
            loss_batch = self._get_loss(x_hat,x)

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
            
            loss[__] = loss_batch.detach().numpy().mean()

        return loss
    
    def _do_test(self):
        """
        Returns:
        loss: A numpy array of shape (test_nb, batch) representing the loss for each test iteration.
        v: A numpy array of shape (test_nb, batch) representing the average membrane potential for each test iteration.
        z: A list of length test_nb, containing numpy arrays of shape (batch, T) representing output spikes.
        fr: A numpy array of shape (test_nb, batch) representing the average firing rate for each test iteration.
        onset: A numpy array of shape (test_nb, batch) representing the onset times for each test iteration, 
        if test_onset is provided, otherwise zeros.
        """
        
        self.train_FLAG = False
        loss = np.zeros((self.par.test_nb,self.par.batch))
        v = np.zeros((self.par.test_nb,self.par.batch))
        fr = np.zeros((self.par.test_nb,self.par.batch))
        onset = np.zeros((self.par.test_nb,self.par.batch))
        z = []

        for __ in range(self.par.test_nb):
            if self.test_onset is not None:
                x,onset_batch = self._get_batch(self.test_data,
                                                self.test_onset)
                onset[__,:] = onset_batch
            else:
                x = self._get_batch(self.test_data)
            self.neuron.state()
            z_batch, v_batch, v_torch = self._forward(x)
            x_hat = torch.einsum("bt,j->bjt",v_torch,self.neuron.w)
            loss_batch = self._get_loss(x_hat,x)

            loss[__,:] = loss_batch.detach().numpy().mean()
            v[__,:] = v_batch.detach().numpy().mean()
            fr[__,:] = self._get_firing_rate(z_batch).mean()
            z.append(z_batch)

        return loss,v,z,fr,onset

    def train(self,log):
        """
        Args:
        log: Log file to save training statistics.

        Returns:
        None

        Updates the following attributes:
        losstrainList: A numpy array to store loss values for training data during each epoch.
        losstestList: A numpy array to store loss values for test data during each epoch.
        frList: A numpy array to store firing rates during each epoch.
        vList: A numpy array to store membrane voltages during each epoch.
        onsetList: A numpy array to store onset times during each epoch for each batch of test data.
        w: A numpy array to store weights during each epoch.
        zList: A list to store outputs of the network during each epoch.
        """
        
        for e in range(self.par.epochs):
            
            loss_train = self._do_train()
            loss_test, v, z, fr, onset = self._do_test()
            
            self.losstrainList[e,:] = loss_train
            self.losstestList[e,:] = loss_test
            self.vList[e,:] = v
            self.frList[e,:] = fr
            self.onsetList[e,:] = onset
            self.w[e,:] = self.neuron.w.detach().clone().numpy()
            self.zList.append(z)
            
            self._save(log,np.mean(loss_train),
                       np.mean(loss_test))
            if e%20 == 0: 
                print('epoch {} \n loss {} \n v {} '.format(
                        e,np.mean(loss_train),np.mean(v)))
                
    def _get_loss(self,x_hat,x):
        """
        Args:
        x_hat: Predicted inputs from the single neuron.
        x: Pre-synaptic input data.

        Returns:
        loss: Calculated loss between x_hat and x.
        """
        
        return loss(x_hat,x)
    
    def _get_firing_rate(self,z):
        """
        Args:
        z: A numpy array of shape (batch_size, T) representing the output spikes.

        Returns:
        A numpy array of shape (batch_size,) representing the firing rate of neurons in Hz.
        """

        return (z.sum(axis=1)/self.par.T)*(1e3/self.par.T)

    def _save(self,log,loss_train,loss_test):
         with open(log,'a') as train:
                train.write('{}, {} \n'.format(loss_train,
                                loss_test))