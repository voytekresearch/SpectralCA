import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import expon
import neurodsp as ndsp
from . import utils


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

    def return_params(self):
        param_dict = {
        'nperseg': self.nperseg,
        'noverlap': self.noverlap,
        'spg_outlierpct': self.spg_outlierpct,
        'max_freq': self.max_freq
        }
        return param_dict

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

        if self.max_freq is not None:
            freq_inds = np.where(self.f_axis<self.max_freq)[0]
            self.f_axis = self.f_axis[freq_inds]
            self.spg = self.spg[:,freq_inds,:]

        self.compute_psd()
        self.compute_scv()

    # calculate the spectrogram
    def compute_spg(self):
        """
        Compute spectrogram of time-series data.
        """
        self.f_axis,self.t_axis,self.spg = sp.signal.spectrogram(self.data,fs=self.fs,nperseg=int(self.nperseg),noverlap=int(self.noverlap))
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
                exp_scale[chan,freq],ks_stats[chan,freq],ks_pvals[chan,freq]=fit_test_exp(self.spg[chan,freq,:],floc=0)
                # param = sp.stats.expon.fit(self.spg[chan,freq,:],floc=0)
                # exp_scale[chan,freq] = param[1]
                # ks_stats[chan,freq], ks_pvals[chan,freq] = sp.stats.kstest(self.spg[chan,freq,:], 'expon', args=param)
        self.exp_scale = exp_scale
        self.ks_pvals = ks_pvals
        self.ks_stats = ks_stats

    def save_spec_vars(self, npz_filename, save_spg=False):
        """ Save the spectral attributes to a .npz file.

        Parameters
        ----------
        npz_filename : str
            Filename of .npz file.
        save_spg: bool (default=False)
            Whether to save all spectrogram.
        """
        param_keys = ['nperseg', 'noverlap','spg_outlierpct', 'max_freq']
        param_vals = [getattr(self, a) for a in param_keys]
        if save_spg:
            np.savez(npz_filename,
             f_axis=self.f_axis,
             psd=self.psd,
             scv=self.scv,
             ks_pvals=self.ks_pvals,
             ks_stats=self.ks_stats,
             exp_scale=self.exp_scale,
             spg = self.spg,
             param_keys=param_keys,
             param_vals=param_vals
            )
        else:
            np.savez(npz_filename,
             f_axis=self.f_axis,
             psd=self.psd,
             scv=self.scv,
             ks_pvals=self.ks_pvals,
             ks_stats=self.ks_stats,
             exp_scale=self.exp_scale,
             param_keys=param_keys,
             param_vals=param_vals
            )

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
        plt.title('Frequency=%.1fHz, p=%.4f' %(self.f_axis[freq_ind],self.ks_pvals[chan,freq_ind]))

    def plot_spectral(self, plot_mean=True, plot_chan=None, plot_color='k'):
        if plot_chan is None:
            plot_chan = np.arange(0,self.numchan)
        titles = ['PSD', 'SCV', 'KS P-Val', 'KS Stats']
        plot_keys = ['psd', 'scv', 'ks_pvals', 'ks_stats']
        for i in range(4):
            plt.subplot(1,4,i+1)
            if plot_mean:
                m, s = _return_meanstd(getattr(self, plot_keys[i]), axis=0)
                plt.fill_between(self.f_axis, m-s, m+s, color=plot_color, alpha=0.5)
                plt.loglog(self.f_axis, m, plot_color)
            else:
                plt.loglog(self.f_axis, getattr(self, plot_keys[i])[plot_chan,:].T, plot_color, alpha=np.max([0.1, 0.75-(np.size(plot_chan)-1)/10]))

            plt.title(titles[i])
            plt.xlim(self.f_axis[1], self.f_axis[-1])
            plt.xlabel('Frequency (Hz)')
            if i is 1: plt.ylim([0.5,5.]) # limit y-axis for scv
            if i is 2: plt.ylim([1e-7,1])


        plt.tight_layout()

def lfpca_load_spec(npz_filename):
    """ Load an .npz file to populate the computed spectral fields of lfpca

    Parameters
    ----------
    npz_filename : str
        Filename of .npz file

    Returns
    -------
    lfpca_obj
        Populated LFPCA object.

    """
    data = np.load(npz_filename)
    analysis_params = dict(zip(data['param_keys'], data['param_vals']))
    data_fields = ['f_axis', 'psd', 'scv', 'ks_pvals', 'ks_stats', 'exp_scale']
    if 'spg' in data.keys():
        # if spectrogram was saved, load as well
        data_fields.append('spg')
    lfpca_obj = LFPCA(analysis_params)
    for df in data_fields:
        setattr(lfpca_obj, df, data[df])
    lfpca_obj.numchan = lfpca_obj.psd.shape[0]
    return lfpca_obj

def fit_test_exp(data, floc=0):
    param = sp.stats.expon.fit(data,floc=floc)
    exp_scale = param[1]
    ks_stat, ks_pval = sp.stats.kstest(data, 'expon', args=param)
    return exp_scale, ks_stat, ks_pval

def compute_BP_HT(data, fs, passband, N_cycles=5, ac_thr=0.05):
    # bandpass filter data
    if passband[0]<=0.:
        # passband starts from 0Hz, lowpass
        data_filt, filt_ker = ndsp.filter(data,fs,'lowpass',f_lo=passband[1],N_cycles=N_cycles,return_kernel=True)
    elif passband[1]<=0.:
        # highpass
        data_filt, filt_ker = ndsp.filter(data,fs,'highpass',f_hi=passband[0],N_cycles=N_cycles,return_kernel=True)
    else:
        # bandpass
        data_filt, filt_ker = ndsp.filter(data,fs,'bandpass',f_lo=passband[0],f_hi=passband[1],N_cycles=N_cycles,return_kernel=True)

    # get effective filter length where autocorrelation drops below the threshold for the last time
    ker_len = np.where(np.abs(utils.autocorr(filt_ker)[1])>=ac_thr)[0][-1]+1

    # get Hilbert transform
    HT = sp.signal.hilbert(data_filt[~np.isnan(data_filt)])

    # amplitude, pad filter edge artifacts with zero
    sig_power = np.ones_like(data)*np.nan
    sig_phase = np.ones_like(data)*np.nan
    sig_power[~np.isnan(data_filt)] = np.abs(HT)**2
    sig_phase[~np.isnan(data_filt)] = np.angle(HT)

    # also return data-valid indices for convenience
    valid_inds = np.where(~np.isnan(data_filt))[0]
    return sig_power, sig_phase, valid_inds, ker_len

def _return_meanstd(data, axis=0):
    return np.mean(data,axis), np.std(data,axis)
