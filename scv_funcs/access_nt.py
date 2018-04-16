import numpy as np
import scipy.io as io


def get_ECoG(data_path, session, chan, indices = [0,0]):
    ''' grabs certain chunks of data

        Parameters
        ----------
        data_path : String
            set the path
        session : integer
            grab the particular session
        chan : list
            grab the particular channels
        indices : list
            specify start and end for each session
            initialized as whole if none is given

        Returns
        -------
        data : 2d array
            return the data
        timefile : file
            return information about the session
    '''
    
    if '%' in data_path:
        data_path = data_path % (session)
        timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)
        if "Session" not in data_path:
            raise DataError("check data!")


    data_block = []

    for c in chan:
        try:
            matfile = io.loadmat(data_path + 'ECoG_ch%d.mat'%(c),
                                 squeeze_me=True)
            data = matfile['ECoGData_ch%d'%(c)]
            if indices == [0,0]:
                indices = [0,len(data)]
            data_block.append(data[indices[0]:indices[1]])
        except Exception as e:
            print('Handling run-time error:', e)
            data = np.empty((len(indices)-1,))
            data[:] = np.nan
            data_block.append(data)

    data_block = np.vstack(data_block)

    return data_block

def get_cond(timefile, start_ind, end_ind):
    ''' grabs indices

        Parameters
        ----------
        timefile : file
            get information about the session
        start_ind : int
            specific condition start
        end_ind : int
            specific condition end

        Returns
        -------
        indices : list
            return the start and end of the data of specific condition
    '''
    if '%' in data_path:
        data_path = data_path % (session)
        timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)
        if "Session" not in data_path:
            raise DataError("check data!")
    start = timefile['ConditionTime'][start_ind]
    end = timefile['ConditionTime'][end_ind]
    indices = [start, end]

    return indices
