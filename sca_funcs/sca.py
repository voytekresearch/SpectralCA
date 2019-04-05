import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.stats import expon
import neurodsp as ndsp
from neurodsp.timefrequency import _hilbert_ignore_nan
from . import utils

import time


# object that has attributes for taking any segment/ time-series data
class SCA:
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
        Initialize SCA object and populate data and analysis parameters.
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

        if self.spg_outlierpct>0.:
            n_discard = int(np.ceil(len(self.t_axis) / 100. * self.spg_outlierpct))
            n_keep = int(len(self.t_axis)-n_discard)
            spg_ = np.zeros((self.numchan,len(self.f_axis),n_keep))
            self.outlier_inds = np.zeros((self.numchan,n_discard))
            for chan in range(self.numchan):
                # discard time windows with high powers, round up so it doesn't get a zero
                self.outlier_inds[chan,:] = np.argsort(np.mean(np.log10(self.spg[chan,:,:]), axis=0))[-n_discard:]
                spg_[chan,:,:] = np.delete(self.spg[chan], self.outlier_inds[chan,:], axis=-1)
            self.spg = spg_
            self.outlier_inds=self.outlier_inds.astype(int)


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
        self.scv = np.std(self.spg, axis=-1) / np.mean(self.spg, axis=-1)

    # # calculate adjacent phase consistency
    # def compute_apc(self):
    #     """
    #     Compute the phase difference between adjacent frequency bins over all
    #     time slices.
    #     """
    # function NFC = neighbor_phase(data, fs, winLen, stepLen)
    # %NFC = neighbor_phase(data, fs, winLen, stepLen)
    #
    #
    # lag = 1;
    # [F Ft Fa] = stft([],data, fs, winLen, stepLen, 400); %prime window size
    # ph = F./abs(F); %normalize fourier amplitude
    # df = fs/winLen;
    # %dph = F(1:end-lag,:,:)./F(1+lag:end,:,:);
    # %keyboard
    # for i=1:(200/df)+1
    #     dph(i,:) = mean((ph(i+1,:,:)./ph(i,:,:)),3);
    # end
    # NFC = abs(dph(1:end-1,:))+abs(dph(2:end,:));



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
    def plot_expfit(self,chan,freq_ind,num_bins=100,plot_cdf=False):
        """
        Plot the histogram of a single frequency, at a single channel.
        """
        spg_slice = self.spg[chan,freq_ind,:]
        rv = expon(scale=sp.stats.expon.fit(spg_slice,floc=0)[1])
        if plot_cdf:
            # plot CDF
            # n,x = np.histogram(spg_slice,normed=True,bins=num_bins)
            # plt.plot(x[:-1], np.cumsum(n), lw=2)
            n, x, _ = plt.hist(spg_slice,bins=num_bins,density=True,cumulative=True, alpha=0.8)
            plt.plot(x, rv.cdf(x), 'k-', lw=2, label='Fit CDF')
        else:
            # plot PDF
            n, x, _ = plt.hist(spg_slice,normed=True,bins=num_bins, alpha=0.8)
            plt.plot(x, rv.pdf(x), 'k-', lw=2, label='Fit PDF')
        #plt.legend()
        plt.xlabel('Power ($V^2/Hz$)', fontsize=14)
        #plt.ylabel('Probability')
        plt.legend(['%.1fHz, p=%.4f' %(self.f_axis[freq_ind],self.ks_pvals[chan,freq_ind])])


    def plot_spectral(self, plot_mean=True, plot_chan=None, plot_color='k', exc_freqs=None):
        if plot_chan is None:
            plot_chan = np.arange(0,self.numchan)

        if exc_freqs is not None:
            p_inds = _freq_to_ind(self.f_axis, exc_freqs)
        else:
            p_inds = np.arange(len(self.f_axis))

        titles = ['PSD', 'SCV', 'KS p-value', 'KS Stats']
        plot_keys = ['psd', 'scv', 'ks_pvals', 'ks_stats']
        for i in range(3):
            ax1=plt.subplot(1,3,i+1)
            if plot_mean:
                m, s = _return_meanstd(getattr(self, plot_keys[i]), axis=0)
                plt.fill_between(self.f_axis[p_inds], m[p_inds]-s[p_inds], m[p_inds]+s[p_inds], color=plot_color, alpha=0.5)
                plt.loglog(self.f_axis[p_inds], m[p_inds], plot_color)
            else:
                plt.loglog(self.f_axis[p_inds], getattr(self, plot_keys[i])[:,p_inds][plot_chan].T, plot_color, alpha=np.max([0.1, 1-(np.size(plot_chan)-1)/5]))

            plt.title(titles[i], fontsize=14)
            plt.xlim(self.f_axis[1], self.f_axis[-1])
            plt.minorticks_off()
            plt.xlabel('Frequency (Hz)', fontsize=14)
            if plot_keys[i] is 'scv':
                plt.yticks([0.5,1,2,4],('0.5','1','2','4')) # limit y-axis for scv
            if i is 2: plt.ylim([1e-7,1])

        plt.tight_layout()

def _freq_to_ind(f_axis, exc_freqs):
    exc_inds = []
    for ef in exc_freqs:
        exc_inds.append(np.where(np.logical_and(f_axis>=ef[0],f_axis<=ef[1]))[0])
    return list(set(np.arange(len(f_axis)))-set(np.concatenate(np.array(exc_inds),axis=0)))

def sca_load_spec(npz_filename):
    """ Load an .npz file to populate the computed spe .ctral fields of SCA

    Parameters
    ----------
    npz_filename : str
        Filename of .npz file

    Returns
    -------
    sca_obj
        Populated sca object.

    """
    data = np.load(npz_filename)
    analysis_params = dict(zip(data['param_keys'], data['param_vals']))
    data_fields = ['f_axis', 'psd', 'scv', 'ks_pvals', 'ks_stats', 'exp_scale']
    if 'spg' in data.keys():
        # if spectrogram was saved, load as well
        data_fields.append('spg')
    sca_obj = SCA(analysis_params)
    for df in data_fields:
        setattr(sca_obj, df, data[df])
    sca_obj.numchan = sca_obj.psd.shape[0]
    return sca_obj

def fit_test_exp(data, floc=0):
    """ Fit and KS test against exponential.

    Parameters
    ----------
    data : type
        Description of parameter `data`.
    floc : type
        Description of parameter `floc`.

    Returns
    -------
    exp_scale : float
        mean parameter of exponential fit.
    ks_stat: float
        KS test statistic.
    ks_pval: float
        KS test p-value.

    """
    param = sp.stats.expon.fit(data,floc=floc)
    exp_scale = param[1]
    ks_stat, ks_pval = sp.stats.kstest(data, 'expon', args=param)
    return exp_scale, ks_stat, ks_pval

def compute_BP_HT(data, fs, passband, N_cycles=5, ac_thr=0.05):
    """ Compute bandpass filtered Hilbert transforms.

    Parameters
    ----------
    data : array, 1D
        Time-series to be filtered.
    fs : float, Hz
        Sampling rate.
    passband : tuple, (f_low, f_high)
        Bandpass pass band, 0 if disregard.
    N_cycles : int, default=5
        Number of cycles of filter.
    ac_thr : float
        Autocorrelation threshold to determine as effective filter length.

    Returns
    -------
    sig_power : array, 1D
        Hilbert power.
    sig_phase : array, 1D
        Hilbert phase.
    valid_inds : array, 1D
        Valid indices of the filtered time-series (non-NaNs).
    ker_len : int, samples
        Effective filter length.
    """
    # bandpass filter data
    if passband[0]<=0.:
        # passband starts from 0Hz, lowpass
        data_filt, filt_ker = ndsp.filter(data,fs,'lowpass',f_lo=passband[1],N_cycles=N_cycles,return_kernel=True)
    elif passband[1]<=0.:
        # highpass
        data_filt, filt_ker = ndsp.filter(data,fs,'highpass',f_hi=passband[0],N_cycles=N_cycles,return_kernel=True)
    else:
        # bandpass
        #data_filt, filt_ker = ndsp.filter(data,fs,'bandpass',f_lo=passband[0],f_hi=passband[1],N_cycles=N_cycles,return_kernel=True)
        data_filt, filt_ker = ndsp.filter(data,fs,'bandpass',fc=passband,N_cycles=N_cycles,return_kernel=True)

    # get effective filter length where autocorrelation drops below the threshold for the last time
    ker_len = np.where(np.abs(utils.autocorr(filt_ker)[1])>=ac_thr)[0][-1]+1

    # use neurodsp Hilbert function to automatically pad to power of 2
    HT = _hilbert_ignore_nan(data_filt,hilbert_increase_N=True)
    sig_power = np.abs(HT)**2
    sig_phase = np.angle(HT)

    # also return data-valid indices for convenience
    valid_inds = np.where(~np.isnan(data_filt))[0]
    return sig_power, sig_phase, valid_inds, ker_len

def _return_meanstd(data, axis=0):
    return np.mean(data,axis), np.std(data,axis)
