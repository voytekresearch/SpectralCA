import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import expon

# object that has attributes for taking any segment/ time-series data
class LFPCA:
    """ Analysis object for time-frequency decomposition of time-series data.
    Attributes:
        data : array, 2D (chan x time)
            Time-series data to be decomposed
        fs : float, Hz
            Sampling frequency of input data
        analysis_params: dict
            'nperseg' : number of samples per segment for STFT computation
            'noverlap' : number of samples overlapping in STFT moving window
            'spg_outlierpct' : percent of windows to consider as outliers
            'max_freq' : maximum frequency to keep, None to keep all
    """

    def __init__(self, analysis_params):
        """
        Initialize LFPCA object and populate data and analysis parameters.
        """
        # parse analysis parameters
        self.nperseg = analysis_params['nperseg']
        self.noverlap = analysis_params['noverlap']
        self.spg_outlierpct = analysis_params['spg_outlierpct']
        self.max_freq = analysis_params['max_freq']

    def populate_ts_data(self,data,fs):
        """
        Populate object with time-series data.
        Data must be 2D (or 1D) array, chan x time
        """
        self.data = data
        self.numchan, self.datalen = data.shape
        self.fs = fs

    def populate_fourier_data(self,data,fs,f_axis,t_axis=None):
        """
        Populate object with Fourier data.
        3D array of time-frequency data (such as spectrogram), where data is of
        dimension [chan x frequency x time].

        For trialed Fourier data, time can be single-trial, in which case t_axis
        would correspond to trial indices.

        Populating with Fourier data automatically computes PSD and SCV.
        """
        numchan, numfreq, numtrials = data.shape
        self.data = []
        self.f_axis = f_axis
        self.spg = data
        self.numchan = numchan
        if t_axis is None:
            self.t_axis = range(numtrials)
        else:
            self.t_axis = t_axis

        self.psd()
        self.scv()

    # calculate the spectrogram
    def compute_spg(self):
        """
        Compute spectrogram of time-series data.
        """
        self.f_axis,self.t_axis,self.spg = sp.signal.spectrogram(self.data,fs=self.fs,nperseg=self.nperseg,noverlap=self.noverlap)
        if self.max_freq is not None:
            freq_inds = np.where(self.f_axis<self.max_freq)[0]
            self.f_axis = self.f_axis[freq_inds]
            self.spg = self.spg[:,freq_inds,:]

    # calculate the psd
    def compute_psd(self):
        """
        Compute the power spectral density using Welch's method
        (mean over spectrogram)
        """
        self.psd = np.mean(self.spg,axis=-1)

    # calculate the spectral coefficient of variation
    def compute_scv(self):
        """
        Compute the spectral coefficient of variation (SCV) by taking standard
        deviation over the mean.
        """
        if self.spg_outlierpct>0.:
            scv_ = np.zeros((self.numchan, len(self.f_axis)))
            spg_ = self.spg
            for chan in range(self.numchan):
                # discard time windows with high powers, round up so it doesn't get a zero
                discard = int(np.ceil(len(self.t_axis) / 100. * self.spg_outlierpct))
                outlieridx = np.argsort(np.mean(np.log10(spg_[chan,:,:]), axis=0))[:-discard]
                scv_[chan, :] = np.std(spg_[chan][:,outlieridx], axis=-1) / np.mean(spg_[chan][:,outlieridx], axis=-1)
            self.scv = scv_
        else:
            self.scv = np.std(self.spg, axis=-1) / np.mean(self.spg, axis=-1)

    # utility so I don't have to write the same 3 lines of code always.
    def compute_all_spectral(self):
        """
        Compute all spectral representations.
        """
        self.compute_spg()
        self.compute_psd()
        self.compute_scv()

    # fit spectrogram slices against exponential distribution and
    # calculate KS-test statistics and p-values
    def compute_KS_expfit(self):
        """
        Compare a given power histogram (per channel & frequency) to the exponential
        null hypothesis using the Kolmogorov-Smirnoff test.

        Return the computed exponential scale value and test statistics.
        """
        exp_scale = np.zeros_like(self.psd)
        ks_pvals = np.zeros_like(self.psd)
        ks_stats = np.zeros_like(self.psd)
        for chan in range(self.numchan):
            for freq in range(len(self.f_axis)):
                param = sp.stats.expon.fit(self.spg[chan,freq,:],floc=0)
                exp_scale[chan,freq] = param[1]
                ks_stats[chan,freq], ks_pvals[chan,freq] = sp.stats.kstest(self.spg[chan,freq,:], 'expon', args=param)
        self.exp_scale = exp_scale
        self.ks_pvals = ks_pvals
        self.ks_stats = ks_stats

    # -------- plotting utilities ------------
    # plotting histogram for a specific channel and frequency and fitting an exp pdf over it
    def plot_expfit(self,chan,freq_ind,num_bins=100):
        """
        Plot the histogram of a single frequency, at a single channel.
        """
        spg_slice = self.spg[chan,freq_ind,:]
#        fig, ax = plt.subplots(1, 1)
        n, x, _ = plt.hist(spg_slice,normed=True,bins=num_bins)
        rv = expon(scale=sp.stats.expon.fit(spg_slice,floc=0)[1])
        plt.plot(x, rv.pdf(x), 'k-', lw=2, label='Fit PDF')
        plt.legend()
        plt.xlabel('Spectral Power')
        plt.ylabel('Probability')
        plt.title('Frequency=%.1f Hz' %self.f_axis[freq_ind])

        # spg_slice = self.spg[chan,freq_ind,:]
        # fig, ax = plt.subplots(1, 1)
        # n, x, _ = ax.hist(spg_slice,normed=True,bins=num_bins)
        # rv = expon(scale=sp.stats.expon.fit(spg_slice,floc=0)[1])
        # ax.plot(x, rv.pdf(x), 'k-', lw=2, label='Fit PDF')
        # plt.legend()
        # plt.title('Frequency=%.1f Hz' %self.f_axis[freq_ind])
