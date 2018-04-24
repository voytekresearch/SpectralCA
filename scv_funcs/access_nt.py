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
            # data is broken for whatever reason
            # this is really not the best solution but we're going to fill it
            # with a channel of white noise
            print('Handling run-time error:', e)
            print('Channel %i is filled in with white noise.'%c)
            data_block.append(np.random.randn(len(data_block[0])))            

    data_block = np.vstack(data_block)

    return data_block

def get_cond(data_path, session, start_ind, end_ind):
    ''' grabs indices

        Parameters
        ----------
        data_path : String
            set the path
        session : integer
            grab the particular session
        start_ind : int
            specific condition start
        end_ind : int
            specific condition end

        Returns
        -------
        indices : list
            return the start and end of the data of specific condition
    '''
    if "Session" not in data_path:
        raise DataError("check data!")
    data_path = data_path % (session)
    timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)
    print(timefile["ConditionLabel"][start_ind], timefile["ConditionLabel"][end_ind])

    return [timefile["ConditionIndex"][start_ind], timefile["ConditionIndex"][end_ind]]
