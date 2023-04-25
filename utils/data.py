import numpy as np

def get_spike_times(par):
    """
    Args:
        par (object): An object containing model parameters.
    Returns:
        spk_times (ndarray): An array of spike times.
    """
    if par.name == 'sequence':
        spk_times = np.linspace(par.Dt/par.dt,(par.Dt/par.dt)*par.N_seq,par.N_seq,dtype=int)
    if par.name == 'selforg':
        spk_times = np.linspace(par.Dt/par.dt,(par.Dt/par.dt)*par.n_in,par.n_in,dtype=int)
        spk_times = np.repeat(spk_times[:,np.newaxis],par.nn,axis=1)
        delays = int(par.delay/par.dt)*np.arange(par.nn)
        spk_times += np.repeat(delays[np.newaxis,:],par.n_in,axis=0)
        if par.network_type == 'random':
            spk_times = spk_times.flatten('F')
            spk_times = np.repeat(spk_times[:,np.newaxis],par.nn,axis=1)
    if par.name == 'multisequence':
        spk_times = np.linspace(par.Dt/par.dt,(par.Dt/par.dt)*par.N_seq,par.N_seq,dtype=int)
        spk_times = np.repeat(spk_times[:,np.newaxis],par.n_afferents,axis=1)
        delays = int(par.delay/par.dt)*np.arange(par.n_afferents)
        spk_times += np.repeat(delays[np.newaxis,:],par.N_seq,axis=0)

    return spk_times

def get_dataset_sequence(par,spk_times):
    """
    Args:
        par (object): An object containing model parameters.
        spk_times (ndarray): An array of spike times.
    Returns:
        x (ndarray): An array representing the dataset.
        onsets (ndarray): An array representing the onset times of the spike trains.
    """
    x = np.zeros((par.batch,par.N,par.T))
    spk_times = np.repeat(spk_times[np.newaxis,:],par.batch,axis=0)
    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.batch,par.N))
        freq = np.repeat(freq[:,:,np.newaxis],par.T,axis=2)
        x[np.random.rand(par.batch,par.N,par.T)<(freq*par.dt/1000)] = 1
    if par.jitter > 0.:
        spk_times += np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,
                                        (par.batch,par.N_seq))
    onsets = None
    if par.onset > 0:
        par.onset = par.T // 2
        onsets = np.random.randint(0,par.onset,par.batch)
        spk_times += np.repeat(onsets[:,np.newaxis],spk_times.shape[1],axis=1)
    for b in range(par.batch):
        for n in range(par.N):    
            if n < par.N_seq: x[b,n,spk_times[b,n]] = 1
            x[b,n,:] = np.convolve(x[b,n,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
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