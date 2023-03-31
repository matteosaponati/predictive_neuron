import json
import types
import numpy as np
import matplotlib.colors as colors

'-----------------------------------------------------------------------------'

def get_dir_results(par):
    'get directory to save results'
    
    if par.name == 'sequence':
        
        path = par.dir_output+('{}/{}/N_seq_{}_N_dist_{}_Dt_{}/'+
                    'freq_{}_jitter_{}_onset_{}/'+
                    'taum_{}_vth_{}_eta_{}/').format(par.package,
                    par.name,par.N_seq,par.N_dist,par.Dt,par.freq,
                    par.jitter,par.onset,par.tau_m,par.v_th,
                    par.eta)
        
    if par.name == 'selforg':

        if par.network_type == 'random':
            path = par.dir_output+('{}/{}/{}/n_in_{}_nn_{}_Dt_{}'+
                    '/freq_{}_jitter_{}/'+
                    'n_afferents_{}_taum_{}_vth_{}_eta_{}/rep_{}/').format(par.package,
                    par.name,par.network_type,par.n_in,par.nn,par.Dt,par.freq,
                    par.jitter,par.n_afferents,par.tau_m,par.v_th,
                    par.eta,par.rep)
        
        else:
            path = par.dir_output+('{}/{}/{}/n_in_{}_nn_{}_delay_{}_Dt_{}'+
                    '/freq_{}_jitter_{}/'+
                    'taum_{}_vth_{}_eta_{}/').format(par.package,
                    par.name,par.network_type,par.n_in,par.nn,par.delay,par.Dt,par.freq,
                    par.jitter,par.tau_m,par.v_th,
                    par.eta)
 
    return path

def get_hyperparameters(par,path):
    
    hyperparameters = vars(par)    
    with open(path+'hyperparameters.json','w') as outfile:
        json.dump(hyperparameters,outfile,indent=4)
    
    return

def get_Namespace_hyperparameters(args):
    
    par = types.SimpleNamespace()
    
    par.name = args['name']
    par.package = args['package']
    
    'training algorithm'
    par.bound = args['bound']
    par.eta = args['eta']
    par.optimizer = args['optimizer']
    par.batch = args['batch']
    par.epochs = args['epochs']
    par.seed = args['seed']
    par.rep = args['rep']

    'initialization'
    par.init = args['init']
    par.init_mean = args['init_mean']
    par.init_a = args['init_a']
    par.init_b = args['init_b']
    par.init_rec = args['init_rec']
    
    'input sequences'
    par.sequence = args['sequence']
    par.Dt = args['Dt']

    par.N_seq = args['N_seq']
    par.N_dist = args['N_dist']
    
    par.network_type = args['network_type']
    par.n_in = args['n_in']
    par.delay = args['delay']
    par.n_afferents = args['n_afferents']
    
    par.freq = args['freq']
    par.jitter = args['jitter']
    par.onset = args['onset']
    
    'model'
    par.dt = args['dt']
    par.tau_m = args['tau_m']
    par.v_th = args['v_th']
    par.tau_x = args['tau_x']

    par.nn = args['nn']
    
    'utils'
    par.dir_output = args['dir_output']
    
    if par.name == 'sequence':
        
        par.T = int(2*(par.Dt*par.N_seq + par.jitter) / (par.dt))
        par.N = par.N_seq+par.N_dist
        if par.onset == 1: par.onset = par.T // 2

    if par.name == 'selforg': 
        
        if par.network_type == 'nearest': 
            
            par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
            par.N = par.n_in+2

        if par.network_type == 'all': 
            
            par.T = int((par.Dt*par.n_in + par.delay*par.nn +  
                     par.jitter + 80)/(par.dt))
            par.N = par.n_in+par.nn

        if par.network_type == 'random':

            par.T = int((par.Dt*par.n_in + par.delay*par.n_in +  
                     par.jitter + 80)/(par.dt))
            par.N_in = par.n_in*par.nn
            par.N = par.N_in+par.nn
        
    return par

def set_device():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  return device

'----------------------------------------------------------------------------------------'
'----------------------------------------------------------------------------------------'

"auxiliary functions plots"

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = colors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp