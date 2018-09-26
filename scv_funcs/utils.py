import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io as io
import scipy.signal as sig
from neurodsp import spectral
import time
import neurodsp as ndsp


def percentile_spectrogram(spg, f_axis, rank_freqs=(8., 12.), pct=(0, 25, 50, 75), sum_log_power=True, show=True):
    """ Compute percentile power spectra using the spectrogram, ranked by power within
    a specific band. Essentially a different way of visualizing correlation between freqs.

    Parameters
    ----------
    spg : array, 2-D (freq x time)
        Spectrogram to be used for PSD computation.
    f_axis : array, 1-D
        Frequency axis of spectrogram.
    rank_freqs : tuple (default=(8,12))
        Frequency band to sum over for ranking.
    pct : tuple (default=(0,25,50,75))
        Left bin edges to bin the spectrogram slices.
    sum_log_power : bool (default=True)
        If true, sum logged power instead of raw power, to counteract 1/f effect.
    show : bool (default=True)
        If true, plot quantile-averaged spectrogram

    Returns
    -------
    power_dgt : array, 1D (1 x time)
        Bin membership of the spectrogram slices.

    """
    f_ind = np.where(np.logical_and(
        f_axis >= rank_freqs[0], f_axis <= rank_freqs[1]))

    if sum_log_power:
        power_vals = np.sum(np.log10(spg[f_ind, :][0]), axis=0)
    else:
        power_vals = np.sum(spg[f_ind, :][0], axis=0)

    bins = np.percentile(power_vals, q=pct)
    power_dgt = np.digitize(power_vals, bins, right=False)
    power_binned = np.asarray(
        [np.mean(spg[:, power_dgt == i], axis=1) for i in np.unique(power_dgt)])
    if show:
        plt.loglog(f_axis, power_binned.T)
        plt.fill_between(rank_freqs, plt.ylim()[0], plt.ylim()[
                         1], facecolor='k', alpha=0.1)
        plt.legend(pct)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')

    return power_dgt, power_binned


def plot_power_examples(data, fs, t_spg, pwr_dgt, rank_freqs, N_cycles=5, N_to_plot=6, power_adj=5, std_YL=True):
    """ Plots examples of time series that fell into the quantile-binned power
    values at the specified frequency. Use in conjunction with percentile_spectrogram.

    Parameters
    ----------
    data : array, 1-D
        Time-series to be plotted.
    fs : float, Hz
        Sampling frequency.
    t_spg : array, 1-D
        Time indices of spectrogram slices.
    pwr_dgt : array, 1-D
        Quantile membership of spectrogram slices, same length as t_spg.
    rank_freqs : tuple, (low, high) Hz
        Band within which total power was ranked by, and will be filtered.
    N_cycles : int, default=5
        Filter order.
    N_to_plot : int, default=6
        Number of examples to plot per quantile.
    power_adj : float
        Adjustment to multiply the filter trace by for better visualization.
        If 0, don't plot filtered data.
    std_YL : bool, default=True
        Standardize y-lim for each subplot for better comparison.

    """
    CKEYS = plt.rcParams['axes.prop_cycle'].by_key()['color']  # grab color rota
    ymin, ymax = 0, 0
    # filter data and multiplier power constant for ease of visualization
    if power_adj:
        data_filt = ndsp.filter(
            data, fs, 'bandpass', f_lo=rank_freqs[0], f_hi=rank_freqs[1], N_cycles=N_cycles) * power_adj
    plot_len = int(fs / 2)
    t_plot = np.arange(-plot_len, plot_len) / fs
    n_bins = len(np.unique(pwr_dgt))
    # loop through bins
    for j in np.unique(pwr_dgt):
        # loop through examples
        for i in range(N_to_plot):
            plt.subplot(N_to_plot, n_bins, (i * n_bins) + j)
            # grab a random window of data that fell within the current power bin
            plot_ind = int(t_spg[np.where(pwr_dgt == j)[0]][np.random.choice(
                len(np.where(pwr_dgt == j)[0]))] * fs)
            y = data[plot_ind - plot_len:plot_ind + plot_len]
            # plot
            plt.plot(t_plot, y - y.mean(), color=CKEYS[j - 1])
            if power_adj:
                plt.plot(
                    t_plot, data_filt[plot_ind - plot_len:plot_ind + plot_len], color=CKEYS[j - 1], alpha=0.5)
            plt.xlim([t_plot[0], t_plot[-1]])
            #plt.title('Ind:%i, Quantile: %i' % (plot_ind, j))
            plt.title('Q%i' % j)
            ymin = min(ymin, plt.ylim()[0])
            ymax = max(ymax, plt.ylim()[1])
    # loop through again and reset y-axis to be the same
    if std_YL:
        for j in np.unique(pwr_dgt):
            for i in range(N_to_plot):
                plt.subplot(N_to_plot, n_bins, (i * n_bins) + j)
                plt.ylim([ymin, ymax])
    plt.tight_layout()


def autocorr(data, max_lag=1000, lag_step=1):
    """ Calculate the signal autocorrelation (lagged correlation)

    Parameters
    ----------
    data : array 1D
        Time series to compute autocorrelation over.
    max_lag : int (default=1000)
        Maximum delay to compute AC, in samples of signal.
    lag_step : int (default=1)
        Step size (lag advance) to move by when computing correlation.

    Returns
    -------
    ac_timepoints : array, 1D
        Time points (in samples) at which correlation was computed.
    ac : array, 1D
        Time lagged (auto)correlation.
    """

    ac_timepoints = np.arange(0, max_lag, lag_step)
    ac = np.zeros(len(ac_timepoints))
    ac[0] = np.sum((data - np.mean(data))**2)
    for ind, lag in enumerate(ac_timepoints[1:]):
        ac[ind + 1] = np.sum((data[:-lag] - np.mean(data[:-lag]))
                             * (data[lag:] - np.mean(data[lag:])))

    return ac_timepoints, ac / ac[0]

def grab_stack_epochs(data, center_idxs, window=(-500,500), axis=-1):
    """ Grab windows of data from a multidimensional array along axis.

    Parameters
    ----------
    data : n-dim array
        Time series data (usually), to be epoched.
    center_idxs : list or array
        List of center indices. Must be iterable.
    window : (int, int), tuple
        Window edge indices relative to center index, which is 0.
        Default (-500, 500)
    axis : int
        Axis along which to grab data. Default -1.

    Returns
    -------
    epoched_data: (n+1)-dim array
        Epoched data concatenated along the last axis.
    """
    # iterate through trial center indices and grab window around it
    trials_app = []
    for i,idx in enumerate(center_idxs):
        beg_idx, end_idx = idx+window[0], idx+window[1]
        if beg_idx>=0 and end_idx<=data.shape[0]:
            trials_app.append(np.take_along_axis(data, np.arange(beg_idx,end_idx), axis=axis))
        else:
            print('Trial %i window exceeds data bounds.'%(i+1))
    # stack in an extra dimension
    return np.stack(trials_app, axis=-1)


# def inst_pwcf(data, fs, frange, n_cycles=3, winLen=1, stepLen=1, logpower=False):
#     """
#     Computes instantaneous frequency & other metrics in a single time series
#     ----- Args -----
#     data : array, 1d
#         time series to calculate center frequency, bandwidth
#         and power for
#
#     fs : float, Hz
#         sampling frequency
#
#     frange : (low, high) Hz
#         frequency range of bandpassed oscillation
#
#     n_cycles : int
#         number of cycles of the FIR filter to use
#         default: 3
#
#     winLen : int, samples
#         size of sliding window to compute stats
#         if 1, return raw estimates
#         default: 1
#
#     stepLen : int, samples
#         step size to advance sliding window
#         default: 1
#
#     logpower : bool
#                 whether to log power
#
#     ----- Returns -----
#     pw: array
#         instantaneous power over time
#
#     cf: array
#         center frequency over time
#
#         scv: array
#                 spectral CV over time (windowed std/mean of power), [] if winLen ==1
#
#     bw: array
#         bandwidth, defined as IQR of instantaneous freq, [] if winLen == 1
#
#     """
#     o_msg = False
#     if o_msg:
#         print('Filtering...')
#
#     filtered = ndsp.filter(data, Fs=fs, pass_type='bandpass',
#                            f_lo=frange[0], f_hi=frange[1], N_cycles=n_cycles, remove_edge_artifacts=False)
#
#     if o_msg:
#         print('Computing Hilbert, Power & Phase...')
#     # compute signal derivatives
#     HT = sp.signal.hilbert(filtered)  # calculate hilbert
#     if logpower:
#         PW = np.log10(abs(HT)**2)  # instantaneous power
#     else:
#         PW = abs(HT)**2  # instantaneous power
#
#     IF = np.diff(np.unwrap(np.angle(HT))) * fs / (
#         2 * np.pi)  # calculate instantaneous frequency
#
#     # moving average & std to compute power, center freq, scv, and bandwidth
#     if winLen == 1:
#         # window size=1, no smoothing
#         return PW, IF, [], []
#
#     else:
#         if o_msg:
#             print('Smoothing...')
#
#         # compute output length & pre-allocate array
#         outLen = int(np.ceil((np.shape(data)[0] - winLen) / float(
#             stepLen))) + 1
#         pw, cf, scv, bw = np.zeros((4, outLen))
#
#         # get smoothed power and coefficient of variation
#         wins =
#         (PW, winLen, stepLen)
#         for ind, win in enumerate(wins):
#             pw[ind] = np.mean(win)
#             scv[ind] = np.std(win) / np.mean(win)
#
#             # get smoothed center freq & bandwidth
#         wins = slidingWindow(IF, winLen, stepLen)
#         for ind, win in enumerate(wins):
#             cf[ind] = np.mean(win)  # smooth center freq
#             bw[ind] = np.diff(np.percentile(win, q=[25, 75]), axis=0)
#
#         return pw, cf, scv, bw


def yield_sliding_window_ts(data, nperseg=1000, noverlap=500):
    """ Return a generator that will iterate through the defined lengths of 1D
    array with window of length nperseg and step length of noverlap; if not a
    perfect divisor, last slice gets remaining data.

    Parameters
    ----------
    data : array, 1D or 2D
        Time series to slide over. If 2D, must be [channel x time]

    nperseg : int, samples, default=1000
        Size of sliding window in number of samples.

    noverlap : int, samples, default=500
        Overlap size of rolling windows.

    Returns
    -------
    Generator with slices of data
        channel x nperseg
    """

    # Pre-compute number of length of output;
    # last slice gets whatever data is remaining
    step_len = int(nperseg-noverlap)
    out_len = int(np.ceil((np.shape(data)[-1] - nperseg) / step_len)) + 1
    # make it rain
    if len(np.shape(data))==1:
        for ind in range(out_len):
            yield data[step_len * ind:nperseg + step_len * ind]
    elif len(np.shape(data))==2:
        for ind in range(out_len):
            yield data[:, step_len * ind:nperseg + step_len * ind]

def yield_sliding_window_pp(event_times, fs, win_len=1., overlap_len=0., end_time=None):
    """ Return a generator that will iterate through a list of event times
    in a sliding window fashion, and return the event times and the absolute
    position (indices) of those event times that fell within each rolling window.

    Parameters
    ----------
    event_times : array, 1D
        Event timestamps (e.g. spiketrain timestamps).
    fs : float, samples
        Sampling rate of the event_times; use to convert spike index into spike
        time in units of time.
    win_len : float, seconds, default=1.0
        Length of rolling window in seconds.
    overlap_len : float, seconds, default=0
        Length of rolling window overlap in seconds.
    end_len : float, seconds, default=None
        Maximum length, in seconds, to slide window to. Handy if trying to match
        a timeseries.
        If None, ends at the last event time.

    Returns
    -------
    Generator that feeds event times and indices.
    Each advance returns a tuple of (event_time, event_index)
    """
    # change event indices into timestamps by dividing by fs
    event_times = event_times / fs
    step_len = win_len - overlap_len
    if end_time is None:
        end_time = event_times[-1]
    # compute how many windows there will be
    out_len = int(np.ceil((end_time - win_len) / step_len)) + 1
    for ind in range(out_len):
        event_inds = np.where((event_times>=ind*step_len) & (event_times< (win_len+ind*step_len)))[0]
        yield event_times[event_inds], event_inds, out_len


def binarize_spiketime(spike_times, len_binned=None, spike_rate=40000., bin_rate=1000.):
    """Takes a vector of spike times and converted to a binarized array, given
    pre-specified spike sampling rate and binned rate.

    Parameters
    ----------
    spike_times : array, 1D
        Array of spike times.
    len_binned : int, default=None
        Length of binarized spike train in samples, can be constrained by
        matching (apriori known) LFP vector length.
        If None, end determined by last spike-time.
    spike_rate : float, default=40000
        Sampling rate of spike stamps; set to 1 if spike time vector contain
        actual time stamps of spikes, instead of indices in sample number.
    bin_rate : floatm default=1000
        Rate at which binarized spike train is sampled at.

    Returns
    -------
    bsp : array, 1D
        Binary array of spikes

    Example
    -------
    Binarize a spike train and compute the rolling window firing rate.

    fr=[]
    bsp = binarize_spiketime(spk_times, spike_rate=20000., bin_rate=1250.)
    for ind, win in enumerate(utils.yield_sliding_window(bsp, win_len=1250, noverlap=1250/2)):
        fr.append(sum(win))
    """
    if len_binned is None:
        # no specified time to truncate, take last spike time as end
        t_end = int(round(spike_times[-1] / spike_rate * bin_rate)) + 1
    else:
        # length of binary vector is predetermined
        t_end = len_binned

    # make binary bins
    bsp = np.zeros(t_end)
    # convert spike index to downsampled index
    inds = (spike_times / spike_rate * bin_rate).astype(int)
    for i in inds:
        # have to loop through indices because there may be overlapping windows
        bsp[i] += 1
    return bsp


"""
#----------------------
"""

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

    plt.xlim(-0.5, nDim - 0.5)
    plt.ylim(nDim - 0.5, -0.5)
    plt.plot([-0.5, nDim - 0.5], [-0.5, nDim - 0.5], 'k-')
    plt.colorbar(fraction=0.046, pad=0.04, ticks=np.linspace(-1, 1, 5))
    plt.tick_params(length=0)
    plt.box()

    # star squares that are significant
    if pv is not None:
        sigInds = np.where(pv < pvThresh)
        plt.scatter(sigInds[1], sigInds[0], s=50, marker='*', c='k')
