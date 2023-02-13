import numpy as np

## for multisequence you just need to change the timing (instead of successive every 2 ms, you put the actual independent sequences)
## and you can just use the same fun

def get_sequence(par,timing,onset=None):

    x = np.zeros((par.N,par.T))

    if par.freq > 0.:
        freq = np.random.randint(0.,par.freq,(par.N))
        freq = np.repeat(freq[:,np.newaxis],par.T,axis=2)
        x[np.random.rand(par.N,par.T)<(freq*par.dt/1000)] = 1

    if par.jitter > 0. :
        timing = timing + np.random.randint(-par.jitter/par.dt,par.jitter/par.dt,par.N)
    if par.onset > 0. :
        timing += onset

    for n in range(par.N):
        x[n,timing[n]] = 1
        x[n,:] = np.convolve(x[n,:],np.exp(-np.arange(0,par.T*par.dt,par.dt)/par.tau_x))[:par.T]  
        
    return x

def get_sequence_SelfOrg(par,timing):

    x  = []
    for n in range(par.nn):
        x.append(get_sequence(par,timing[n]))

    return np.stack(x,axis=2)