import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.signal as sig
from neurodsp import spectral
import time
import neurodsp as ndsp



# def scv_filt(x, Fs, ):

# 	ndsp.filter()

# def grid_fit(f_axis, data, min_len=5, max_len=50, fit_constrain=(None,None), model='ols'):
# 	"""
# 	Moving window linear fitting of PSD, SCV,
# 	"""

# def rolling_linfit():




def inst_pwcf(data, fs, frange, n_cycles=3, winLen=1, stepLen=1, logpower=False):
    """
    Computes instantaneous frequency & other metrics in a single time series
    ----- Args -----
    data : array, 1d
        time series to calculate center frequency, bandwidth
        and power for

    fs : float, Hz
        sampling frequency

    frange : (low, high) Hz
        frequency range of bandpassed oscillation

    n_cycles : int
        number of cycles of the FIR filter to use
        default: 3

    winLen : int, samples
        size of sliding window to compute stats
        if 1, return raw estimates
        default: 1

    stepLen : int, samples
        step size to advance sliding window
        default: 1

    logpower : bool
		whether to log power

    ----- Returns -----
    pw: array
        instantaneous power over time

    cf: array
        center frequency over time

	scv: array
		spectral CV over time (windowed std/mean of power), [] if winLen ==1

    bw: array
        bandwidth, defined as IQR of instantaneous freq, [] if winLen == 1

    """
    o_msg = False
    if o_msg:
        print('Filtering...')

    filtered = ndsp.filter(data, Fs=fs, pass_type='bandpass',
    	f_lo=frange[0], f_hi=frange[1], N_cycles=n_cycles, remove_edge_artifacts=False)

    if o_msg:
        print('Computing Hilbert, Power & Phase...')
    # compute signal derivatives
    HT = sp.signal.hilbert(filtered)  # calculate hilbert
    if logpower:
    	PW = np.log10(abs(HT)**2) # instantaneous power
    else:
    	PW = abs(HT)**2 # instantaneous power

    IF = np.diff(np.unwrap(np.angle(HT))) * fs / (
        2 * np.pi)  # calculate instantaneous frequency

    # moving average & std to compute power, center freq, scv, and bandwidth
    if winLen == 1:
        # window size=1, no smoothing
        return PW, IF, [], []

    else:
        if o_msg:
            print('Smoothing...')

        # compute output length & pre-allocate array
        outLen = int(np.ceil((np.shape(data)[0] - winLen) / float(
            stepLen))) + 1
        pw, cf, scv, bw = np.zeros((4, outLen))

        # get smoothed power and coefficient of variation
        wins = slidingWindow(PW, winLen, stepLen)
        for ind, win in enumerate(wins):
            pw[ind] = np.mean(win)
            scv[ind] = np.std(win)/np.mean(win)

            # get smoothed center freq & bandwidth
        wins = slidingWindow(IF, winLen, stepLen)
        for ind, win in enumerate(wins):
            cf[ind] = np.mean(win)  # smooth center freq
            bw[ind] = np.diff(np.percentile(win, q=[25, 75]), axis=0)

        return pw, cf, scv, bw


def slidingWindow(data, winLen=1000, stepLen=500):
    """
    Returns a generator that will iterate through
    the defined lengths of 1D array with window of
    length winLen and step length of stepLen;
    if not a perfect divisor, last slice gets remaining data
    --- Args ---
    data : array, 1d
        time series to slide over

    winLen : int, samples
        size of sliding window
        default: 1000

    stepLen : int, samples
        step size of window
        default: 500

    --- Return ---
    generator with slices of data
        channel x winLen
    """

    # Pre-compute number of length of output
    # last slice gets whatever data is remaining
    outLen = int(np.ceil((np.shape(data)[0] - winLen) / float(stepLen))) + 1
    # make it rain
    for ind in range(outLen):
        yield data[stepLen * ind:winLen + stepLen * ind]


def bin_spikes(spikeTimes, binnedLen=-1, spkRate=40000., binRate=1000.):
    """
    Takes a vector of spike times and converted to a binarized array, given
    pre-specified spike sampling rate and binned rate
    example use:
        bsp = bin_spikes(spk_times, spkRate=20000., binRate=1250.)
        for ind, win in enumerate(utils.slidingWindow(bsp, winLen=1250, stepLen=1250/2)):
            smob[ind] = sum(win[0])
    --- Args ---
    spikeTimes : array, 1d
        list of spike times

    binnedLen : int
        length of binarized spike train in samples, can be constrained
        by matching LFP vector length
        default: -1, end determined by last spike-time

    spkRate : float
        sampling rate of spike stamps, 1. if spike time vector contain
        actual time stamps of spikes, instead of indices
        default: 40000.

    binRate : float
        rate at which binarized spike train is sampled at
        default: 1000.

    --- Return ---
    binary array of spikes (float)
    """
    if binnedLen == -1:
        # no specified time to truncate, take last spike time as end
        t_end = int(round(spikeTimes[-1] / spkRate * binRate)) + 1
    else:
        # length of binary vector is predetermined
        t_end = binnedLen

    # make binary bins
    bsp = np.zeros(t_end)
    # convert spike index to downsampled index
    inds = spikeTimes / spkRate * binRate
    # truncate spikes to match end time & make int

    for i in inds:
        bsp[i] += 1
    # bsp[inds[inds < t_end].astype(int)] += 1
    return bsp


def smooth_events(eventTimes, values=None, fs=1., winLen=1., stepLen=0.5):
    """
    Takes a vector of event times (or samples) and compute rolling window
    event count (or average ) over the events.
    --- Args ---
    eventTimes: array, 1d
        array of event times (or sample indices)

    values: array, 1d
        array of event values
        default: None

    fs: float
        sampling rate of eventTimes
        default: 1. i.e. represents time in seconds

    winLen: float
        time length of window
        default: 1 second

    stepLen: float
        time length of stepping
        default: 0.5 seconds

    --- Return ---
    array of smoothed values (float)

    """
    # change event indices into timestamps by dividing by fs
    eventTimes = eventTimes * 1. / fs
    outLen = int(np.ceil((eventTimes[-1] - winLen) / float(stepLen))) + 1
    smoothed = np.zeros(outLen)
    if values is None:
        # no values attached, just count how many occurrences (probably spikes)
        for ind in range(outLen):
            smoothed[ind] = np.shape(eventTimes[(eventTimes >= ind * stepLen) &
                                                (eventTimes < winLen + ind *
                                                 stepLen)])[0]
    else:
        # average the values
        for ind in range(outLen):
            smoothed[ind] = np.mean(values[(eventTimes >= ind * stepLen) & (
                eventTimes < winLen + ind * stepLen)])

    return smoothed


def corrcoefp(matrix):
    """
    Takes in matrix and calculates column-wise pair correlation and p-value
    Does what matlab's corr does
    copied from scipy.stats.pearsonr and adapted for 2D matrix
    --- Args ---
    matrix: array, 2D (dim x sample)
        array to calculate correlation over, every pair of dimensions

    --- Return ---
    r: array, 2D
        correlation matrix

    p: array, 2D
        p-value matrix
    """
    r = np.corrcoef(matrix)
    df = matrix.shape[1] - 2
    t_squared = r**2 * (df / ((1.0 - r) * (1.0 + r)))
    p = sp.special.betainc(0.5 * df, 0.5, df / (df + t_squared))
    np.fill_diagonal(p, 1.0)

    return r, p


def corr_plot(C, labels=None, pv=None, pvThresh=0.01, cmap='RdBu', bounds=None):
    """
    Takes in a correlation matrix and draws it, with significance optional
    --- Args ---
    C: array, 2D square matrix
        correlation matrix to be plotted

    labels: list of strings
        label for each row in correlation matrix
        must match dimension of C

    pv: 2d square matrix
        significance matrix, should match dims of C
        default: None

    pvThresh: float
        threshold value to draw significance stars
        default: 0.01

    cmap: str
        colormap of plotted matrix
        default: 'RdBu' - redblue

    bounds: [val1, val2]
        bounds on the colormap
        default: None

    """
    # fill diagonals to zero
    np.fill_diagonal(C, 0.)
    # define color bounds
    if bounds is None:
        vmin = -1.
        vmax = 1.
    else:
        vmin, vmax = bounds

    nDim = np.shape(C)[0]
    # draw the square
    plt.imshow(C, interpolation='none', cmap='RdBu', vmin=vmin, vmax=vmax)
    if labels is not None:
        plt.xticks(np.arange(len(labels)), labels)
        plt.yticks(np.arange(len(labels)), labels)


    plt.xlim(-0.5, nDim-0.5)
    plt.ylim(nDim-0.5, -0.5)
    plt.plot([-0.5, nDim - 0.5], [-0.5, nDim - 0.5], 'k-')
    plt.colorbar(fraction=0.046, pad=0.04, ticks=np.linspace(-1, 1, 5))
    plt.tick_params(length=0)
    plt.box()

    # star squares that are significant
    if pv is not None:
        sigInds = np.where(pv < pvThresh)
        plt.scatter(sigInds[1], sigInds[0], s=50, marker='*', c='k')
