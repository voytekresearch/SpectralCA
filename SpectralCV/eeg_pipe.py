# Setup 
import numpy as np
import scipy as sp
import scipy.io as io
import scipy.signal as sig
import math as math
import random 
from scipy import integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py

def getEEGdata(mydata_path, sub):
    data_path = mydata_path
    matfile = io.loadmat(data_path + '%d.mat'%(sub), squeeze_me=True)
    data = matfile['oz_rest_data']
    return data

def sublist():
    return np.concatenate((np.arange(1000,1015), np.arange(2000,2014)),axis=0)

def getFreq(data, fs, nperseg, noverlap):
    f_axis, f_time, spg = sig.spectrogram(data, fs=fs, nperseg=fs, noverlap=fs)
    return len(f_axis)

def humanBread(sublist,fs,nperseg,noverlap):
    bread = np.zeros((len(sublist), 501))
    for i, sub in enumerate(sublist):
        data = getEEGdata(data_path,sub)
        f_axis, f_time, spg = sig.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        bread[i][:] = ep.scv(spg)
    return bread