import numpy as np
#import neurodsp as ndsp
import matplotlib.pyplot as plt
from scipy.stats import expon
import scipy as sp

# object that has attributes for taking any segment/ time-series data
class LFPCA:

    # segment also include attributes of fs, nperseg, and noverlap
    def __init__(self,data,fs,nperseg,noverlap=0,spg_outlierpct=0, max_freq=None):
        # data has to be 2D array, chan x time
        self.data = data
        self.numchan, self.datalen = data.shape
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.spg_outlierpct = spg_outlierpct
        self.max_freq = max_freq


    # calculate the spg of a given channel
    def compute_spg(self):
        self.f_axis,self.t_axis,self.spg = sp.signal.spectrogram(self.data,fs=self.fs,nperseg=self.nperseg,noverlap=self.noverlap)
        if self.max_freq is not None:
            freq_inds = np.where(self.f_axis<self.max_freq)[0]
            self.f_axis = self.f_axis[freq_inds]
            self.spg = self.spg[:,freq_inds,:]

    # calculate the psd
    def compute_psd(self):
        self.psd = np.mean(self.spg,axis=-1)

    # calculate the spectral coefficient of variation
    def compute_scv(self):
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

    # calculate exponential scaling parameter
    def exp_scale(self,freq):
        scale = sp.stats.expon.fit(self.spg[freq],floc=0)
        return scale

    # plotting histogram for a specific channel and frequency and fitting an exp pdf over it
    def plot_expfit(self,chan,freq_ind,num_bins=100):
        spg_slice = self.spg[chan,freq_ind,:]
        fig, ax = plt.subplots(1, 1)
        n, x, _ = ax.hist(spg_slice,normed=True,bins=num_bins)
        rv = expon(scale=sp.stats.expon.fit(spg_slice,floc=0)[1])
        ax.plot(x, rv.pdf(x), 'k-', lw=2, label='Frozen PDF')
        plt.legend()
        plt.title('Frequency=%.1f Hz' %self.f_axis[freq_ind])

    # fit spectrogram slices against exponential distribution and
    # calculate KS-test statistics and p-values
    def compute_KS_expfit(self):
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
