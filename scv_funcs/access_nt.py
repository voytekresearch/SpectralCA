import matplotlib.pyplot as plt
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
            data_block.append(np.random.randn(indices[1]-indices[0]))

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
        raise ValueError("Session not in data path.")
    data_path = data_path % (session)
    timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)
    print(timefile["ConditionLabel"][start_ind] + ' to ' + timefile["ConditionLabel"][end_ind])

    return [timefile["ConditionIndex"][start_ind], timefile["ConditionIndex"][end_ind]]

def ctx_viz(ctx_file, data=None, chans=None, ms=20.):
    """ Plots the monkey ECoG grid over cortex, with intensity optionally
    defined by a data vector.

    Parameters
    ----------
    ctx_file : str
        Neurotycho .mat file path for cortex image.
    data : None or 1D array
        If array, electrode grid is colored according to values in array.
    chans : None or 1D array
        Channel indices to plot. If None, defaults to 0:len(data).
    ms : float
        Marker size.
    """

    ctx_mat = io.loadmat(ctx_file, squeeze_me=True)
    plt.imshow(ctx_mat['I'])
    if data is None:
        # just plot the
        plt.scatter(ctx_mat['X'],ctx_mat['Y'], marker='o', c='w', s=ms)
    else:
        if chans is None:
            # # automatically fill in channel numbers assuming same order as data array
            chans = np.arange(len(data))
        plt.scatter(ctx_mat['X'][chans],ctx_mat['Y'][chans], marker='o', s=ms, c=data, cmap='Blues')
        cbar = plt.colorbar(fraction=0.05)
        cbar.set_ticks([min(data),max(data)])

    plt.box('off')
    plt.xlim([50, 950])
    plt.ylim([1200, 40])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout()
