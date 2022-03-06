"""
----------------------------------------------
Copyright (C) Vinck Lab
-add copyright-
----------------------------------------------
"models.py"
Predictive processes at the single neuron level 

Author:
    
    Matteo Saponati
    Vinck Lab, Ernst Struengmann Institute for Neuroscience
    in cooperation with the Max-Planck Society
----------------------------------------------
"""

import torch
import torch.nn.functional as F

def get_sequence(par,timing):

    prob = par.freq*par.dt
    mask = torch.rand(par.batch,par.T,par.N).to(par.device)
    x_data = torch.zeros(par.batch,par.T,par.N).to(par.device)
    x_data[mask<prob] = 1.0
    
    x_data[:,timing,range(par.N)] = 1
    
    # for b in range(par.batch):
    #     # timing_err = np.array(timing) + np.random.randint(0,2,par.N)
    #     x_data[b,timing.tolist(),range(par.N)]
        
    filter = torch.tensor([(1-par.dt/par.tau_x)**(par.T-i-1) 
                                for i in range(par.T)]).view(1,1,-1).float()
    x_data = F.conv1d(x_data.permute(0,2,1),filter.expand(par.N,-1,-1),
                         padding=par.T,groups=par.N)[:,:,1:par.T+1]
    
    return x_data.permute(0,2,1)