"""
eeg_pipe.py
pipeline for eeg data to do the coefficient of variation of power spectral density.

"""

import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import scv

def getEEGdata(mydata_path, channel_data, sub):
    """
    grabbing the EEG data 
    
    Parameters
    ----------
    mydata_path : String
        set the path
    channel_data : String
        grab the particular channel
    sub : integer
        grab the particular subject
    
    Returns
    -------
    data : array-like 1d
        return the data.
        
    """
    data_path = mydata_path
    matfile = io.loadmat(data_path + '%d.mat'%(sub), squeeze_me=True)
    data = matfile[channel_data]
    return data

def sublist():
    """
    Returns
    -------
    subject list
    
    """
    return np.concatenate((np.arange(1000,1015), np.arange(2000,2014)),axis=0)

def getFreq(data, fs, nperseg, noverlap):
    """
    getting the amount of frequency window for the x-axis.
        
    Parameters
    ----------
    data : array-like 1d
        the data.
    fs : integer
        frequency.
    nperseg : integer
        desired window per segment.
    noverlap : integer
        desired overlapping of window.
    
    Returns
    -------
    length of amount of frequency window.
        
    """
    f_axis, f_time, spg = sig.spectrogram(data, fs=fs, nperseg=fs, noverlap=fs)
    return len(f_axis)

def humanBread(sublist, fs, nperseg, noverlap):
    """
    getting the amount of frequency window for the x-axis.
    
    Parameters
    ----------
    sublist : list
        subject list
    fs : integer
        frequency.
    nperseg : integer
        desired window per segment.
    noverlap : integer
        desired overlapping of window.
    
    Returns
    -------
    bread : array-like 3d
           return the data- frequency x subject        
    """
    bread = np.zeros((len(sublist), getfreq(data, fs, nperseg, noverlap)))
    for i, sub in enumerate(sublist):
        data = getEEGdata(data_path,sub)
        f_axis, f_time, spg = sig.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        bread[i][:] = scv.scv(spg)
    return bread