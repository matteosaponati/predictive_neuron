import numpy as np

def get_spike_times(par) -> np.ndarray:
    """ Generate input spike times given a simulation type 

    Args:
        par (object): An object containing model parameters.
    Returns:
        spk_times (np.ndarray): An array of spike times.
    """

    ## the time interval between sequential spikes in units of dt
    time_bin = par.Dt / par.dt

    if par.name == 'sequence':
        spk_times = time_bin * np.arange(1, par.N_seq + 1)

    elif par.name == 'selforg':
        spk_times = np.arange(1, par.n_in + 1) * time_bin
        spk_times = np.tile(spk_times[:, np.newaxis], (1, par.nn))
        delays = np.arange(par.nn) * (par.delay / par.dt)
        spk_times = spk_times + np.repeat(delays[np.newaxis,:], par.n_in, axis = 0)
        if par.network_type == 'random':
            spk_times = np.tile(spk_times.flatten('F')[:, np.newaxis], (1, par.nn))
    
    elif par.name == 'multisequence':
        spk_times = np.arange(1, par.N_seq + 1) * time_bin
        spk_times = np.tile(spk_times[:, np.newaxis], (1, par.n_afferents))
        delays = np.arange(par.n_afferents) * (par.delay / par.dt)
        spk_times = spk_times + np.repeat(delays[np.newaxis,:], par.n_in, axis = 0)

    return spk_times

def get_dataset_sequence(par, spk_times) -> tuple[np.ndarray, np.ndarray]:
    """ Generates a dataset given a set of spike times.

    Args:
        par (Namespace): an object containing model parameters:
            - batch (int): number of batches.
            - N (int): number of pre-synaptic neurons.
            - T (int): total time steps.
            - freq (float): frequency of background firing.
            - dt (float): timestep.
            - jitter (float): random jitter on spike times.
            - N_seq (int): number of sequences (for multisequence inputs).
            - onset (int): set of random onset of input sequence
            - tau_x (float): time constant to low-pass filter input spikes
        
        spk_times (ndarray): An array of spike times.

    Returns:
        x (ndarray): An array representing the dataset.
            - Shape: (batch, N, T)
        onsets (ndarray): An array representing the onset times of the spike trains.
            - Shape: (batch,)
    """

    x = np.zeros((par.batch, par.N, par.T))
    spk_times = np.tile(spk_times, (par.batch, 1))

    ## (1) background Homogenenous Poisson firing
    if par.freq > 0.:
        freq = (par.dt / 1e3) * np.random.randint(0., par.freq, (par.batch, par.N))
        x = (np.random.rand(par.batch, par.N, par.T) < (freq[:, np.newaxis])).astype(np.float32)

    ## (2) random jitter in input spike trains
    if par.jitter > 0.:
        spk_times = spk_times + np.random.randint(-par.jitter / par.dt,
                                                  par.jitter / par.dt,
                                                  (par.batch,par.N_seq))
        
    ## (3) Random sequence onset
    onsets = None
    if par.onset > 0:
        par.onset = par.T // 2
        onsets = np.random.randint(0, par.onset, par.batch)
        spk_times += np.repeat(onsets[:,np.newaxis],spk_times.shape[1],axis=1)
    
    ## create filtered spike times
    for b in range(par.batch):
        for n in range(par.N):    
            if n < par.N_seq: x[b, n, spk_times[b, n]] = 1
            
    low_pass_filter = np.exp(-np.arange(par.T) * par.dt / par.tau_x)
    x = np.apply_along_axis(lambda m: np.convolve(m, low_pass_filter, mode='full')[: par.T], 
                            axis = 2, arr = x)
    
    return x, onsets

def get_dataset_selforg(par,spk_times):
    """
    Args:
        par (object): An object containing model parameters.
        spk_times (ndarray): An array of spike times.
    Returns:
        x (ndarray): An array representing the dataset.
    """
    x = np.zeros((par.batch,par.n_in,par.nn,par.T))
    spk_times = np.repeat(spk_times[np.newaxis,:],par.batch,axis=0)
    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.batch,par.n_in,par.nn))
        freq = np.repeat(freq[:,:,:,np.newaxis],par.T,axis=3)
        x[np.random.rand(par.batch,par.n_in,par.nn,par.T)<(freq*par.dt/1000)] = 1
    if par.jitter > 0.:
        jitter = np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,
                                        (par.batch,par.n_in))
        spk_times += np.repeat(jitter[:,:,np.newaxis],par.nn,axis=2)
    for b in range(par.batch):
        for n in range(par.n_in):
            for nn in range(par.nn):    
                x[b,n,nn,spk_times[b,n,nn]] = 1
                x[b,n,nn,:] = np.convolve(x[b,n,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
    return x

def get_dataset_random(par,spk_times):
    """
    Args:
        par (object): An object containing model parameters.
        spk_times (ndarray): An array of spike times.
    Returns:
        x (ndarray): An array representing the dataset.
    """
    x = np.zeros((par.batch,par.N_in,par.nn,par.T))
    spk_times = np.repeat(spk_times[np.newaxis,:],par.batch,axis=0)
    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.batch,par.N_in,par.nn))
        freq = np.repeat(freq[:,:,:,np.newaxis],par.T,axis=3)
        x[np.random.rand(par.batch,par.N_in,par.nn,par.T)<(freq*par.dt/1000)] = 1
    if par.jitter > 0.:
        jitter = np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,
                                        (par.batch,par.N_in))
        spk_times += np.repeat(jitter[:,:,np.newaxis],par.nn,axis=2)
    for b in range(par.batch):
        for n in range(par.N_in):
            for nn in range(par.nn):    
                x[b,n,nn,spk_times[b,n,nn]] = 1
                x[b,n,nn,:] = np.convolve(x[b,n,nn,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
    return x

def get_dataset_multisequence(par,spk_times):
    """
    Args:
        par (object): An object containing model parameters.
        spk_times (ndarray): An array of spike times.
    Returns:
        x (ndarray): An array representing the dataset.
    """
    x = np.zeros((par.batch,par.N,par.T))
    spk_times = np.repeat(spk_times[np.newaxis,:],par.batch,axis=0)
    N = np.repeat(np.arange(0,par.N_seq,dtype=int)[:,np.newaxis],
                  par.n_afferents,axis=1)
    N +=  np.repeat(par.N_seq*np.arange(0,par.n_afferents,dtype=int)[np.newaxis,:],
                   par.N_seq,axis=0) 
    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.batch,par.N))
        freq = np.repeat(freq[:,:,np.newaxis],par.T,axis=2)
        x[np.random.rand(par.batch,par.N,par.T)<(freq*par.dt/1000)] = 1
    if par.jitter > 0.:
        spk_times += np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,
                                        (par.batch,par.N_seq,par.n_afferents))
    for b in range(par.batch):
        j = np.random.randint(0,par.n_afferents)
        for n in range(par.N):    
            if n < par.N_seq:
                x[b,N[n,j],spk_times[b,n,j]] = 1
            x[b,n,:] = np.convolve(x[b,n,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
        
    return x