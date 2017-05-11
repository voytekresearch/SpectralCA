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

def getECoGdata(mydata_path, session_num, chan):
    if "Session" not in mydata_path:
        raise DataError("check data!")
    data_path = mydata_path % (session_num)
    matfile = io.loadmat(data_path + 'ECoG_ch%d.mat'%(chan), squeeze_me=True)
    timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)
    data = matfile['ECoGData_ch%d'%(chan)]
    return data

def getTimeFile(mydata_path, session_num):
    if "Session" not in mydata_path:
        raise DataError("check data!")
    data_path = mydata_path % (session_num)
    timefile = io.loadmat(data_path + 'Condition.mat', squeeze_me=True)
    return timefile

def getFreq(data, fs, nperseg, noverlap):
    f_axis, f_time, spg = sig.spectrogram(data, fs=fs, nperseg=fs, noverlap=fs)
    return len(f_axis)

def conditionInfo(timefile):
    return(timefile['ConditionLabel'])

def getSP(data, start, end, fs, nperseg, noverlap):
    f_axis, f_time, spg = sig.spectrogram(data, fs = fs, nperseg = nperseg, noverlap = noverlap)
    SP = spg[:,np.intersect1d(np.where(f_time>start),np.where(f_time<end))]
    return SP

def getStart(timefile, start_ind):
    return timefile['ConditionTime'][start_ind]

# getting the end index
def getEnd(timefile, end_ind):
    return timefile['ConditionTime'][end_ind]

# getting the cv value
def scv(SP):
    return ((np.std(SP,axis=1)/np.mean(SP,axis=1)))

def psd(SP):
    return (np.mean(SP,axis=1))

# plotting the scv
def plotScv(SP):
    plt.loglog(np.std(SP,axis=1)/np.mean(SP,axis=1))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('CV')

# plotting the psd
def pltPsd(data, start, end, session_num, fs, nperseg, noverlap):
    f_axis, f_time, spg = sig.spectrogram(data, fs, nperseg, noverlap)
    plt.loglog(np.mean(spg[:,np.intersect1d(np.where(f_time>timefile['ConditionTime'][start]),np.where(f_time<timefile['ConditionTime'][end]))],axis=1))

# grabbing all the data that is less than 1 to be placed into a dictionary
# note the index of the channels got shifted by 1 because of 0 indexing
# keys are channels and values are how many data are under 1
def createLess(bread, dict_name,  breadSlice):
    for c in range(128):
        for i in range(bread[breadSlice][1].size): #range size of the data
            if bread[breadSlice][c][i] < 1:
                if (c+1) not in list(dict_name.keys()):
                    dict_name[int(c+1)] = 1
                else:
                    dict_name[int(c+1)] = int(dict_name.get(int(c+1)) + 1)
    return dict_name


def less_dict(h5_file, name, shorth5path, breadslice):
    with h5py.File(h5_file, 'r') as h5:
        bread = h5[shorth5path]
        createLess(bread, dictname, breadslice)
    return dictname

def monkeyBread(data_path, chan, fs, nperseg, noverlap):
    cond = 5 #amount of slices (thickness)
    data = getECoGdata(data_path, 1, 1) 
    f_axis, f_time, spg = sig.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
    frequency = len(f_axis)
    bread = np.zeros((cond, chan, frequency))
    session1(bread=bread, data_path=data_path, chan=chan, fs=fs, nperseg=nperseg, noverlap=noverlap)
    session2(bread=bread, data_path=data_path, chan=chan, fs=fs, nperseg=nperseg, noverlap=noverlap)
    session3(bread=bread, data_path=data_path, chan=chan, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return bread

def session1(bread, data_path, chan, fs, nperseg, noverlap):
# session 1
    for i in range(1,chan):
        #grabbing session 1 channel i
        data = getECoGdata(data_path, 1, i)
        #grabbing TimeFile from session 1 to set the indices for each condition
        tf1 = getTimeFile(data_path, 1)    
        # AwakeEyesOpened
        s1start1 = getStart(tf1, 0)
        s1end1 = getEnd(tf1, 1)
        s1SP1 = getSP(data, s1start1, s1end1, fs, nperseg, noverlap)
        bread[0][i-1][:] = scv(s1SP1)
        #AwakeEyesClosed
        s1start2 = getStart(tf1, 2)
        s1end2 = getEnd(tf1, 3)
        s1SP2 = getSP(data, s1start2, s1end2, fs, nperseg, noverlap)
        bread[1][i-1][:] =scv(s1SP2)

def session2(bread, data_path, chan, fs, nperseg, noverlap):
# session 2
    for i in range(1,chan):
	    if('0730PF' in data_path and i == 63):
            	bread[2][i-1][:] = "Nan"
            	bread[3][i-1][:] = "Nan"
	    else:
        	#grabbing session 2 channel i
        	data = getECoGdata(data_path, 2, i)
        	#grabbing TimeFile from session 2 to set the indices for each condition
        	tf2 = getTimeFile(data_path, 2)    
        	#Anesthetized-Start
        	s2start1 = getStart(tf2, 1)
        	s2end1 = getEnd(tf2, 2)
        	s2SP1 = getSP(data, s2start1, s2end1, fs, nperseg, noverlap)
        	bread[2][i-1][:] = scv(s2SP1)
        	#RecoveryEyesClosed
        	s2start2 = getStart(tf2, 3)
        	s2end2 = getEnd(tf2, 4)
        	s2SP2 = getSP(data, s2start2, s2end2, fs, nperseg, noverlap)
        	bread[3][i-1][:] = scv(s2SP2)

def session3(bread, data_path, chan, fs, nperseg, noverlap):
# session 3
    for i in range(1,chan):
        #grabbing session 3 channel i
        data = getECoGdata(data_path, 3, i)
        #grabbing TimeFile from session 3 to set the indices for each condition
        tf3 = getTimeFile(data_path, 3)    
        # RecoveryEyesOpened
        s3start = getStart(tf3, 0)
        s3end = getEnd(tf3, 1)
        s3SP = getSP(data, s3start, s3end, fs, nperseg, noverlap)
        bread[4][i-1][:] = scv(s3SP)