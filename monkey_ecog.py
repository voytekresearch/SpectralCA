import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
from neurodsp import spectral

def get_ECoG(data_path, session, chan, indices):

    data_path = data_path % (session)
    if "Session" not in data_path:
        raise DataError("check data!")
    
    timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)

    data_block = []
    
    for c in chan:
        try:
            matfile = io.loadmat(data_path + 'ECoG_ch%d.mat'%(c), squeeze_me=True)
            data = matfile['ECoGData_ch%d'%(c)]
            data_block.append(data[indices[0]:indices[1]])
        except Exception as e:
            print('Handling run-time error:', e)
            data = np.empty((len(indices)-1,))
            data[:] = np.nan
            data_block.append(data)
    

    return data_block, timefile

def get_cond(timefile, start_ind, end_ind):
	start = timefile['ConditionTime'][start_ind]
	end = timefile['ConditionTime'][end_ind]
	indices = [start, end]
	return indices


