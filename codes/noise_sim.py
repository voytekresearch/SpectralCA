import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn
import sys
from neurodsp import spectral
import neurodsp as ndsp

def noise_sim(color, numtrs, with_osc = False, oFreq = 20.22, oAmp = 20):
	fs = 1000.
	t = np.arange(0.,600.,1./fs)
	ker = np.exp(-50*np.arange(0,500)/1000.)

	# steady sinusoidal oscillation
	osc = np.sin(2*np.pi*oFreq*np.arange(0,len(t))/fs)*1.

	# initialize array
	SCVnoise = np.zeros((numtrs,501))
	if color == 'white':
		for tr in range(0,numtrs):
	    	# white noise
	    	x = np.random.normal(size=(len(t)))
	    	if with_osc == True:
	    		f, SCV = spectral.scv(x+osc, fs, noverlap=0)
	    	else:
	    		f, SCV = spectral.scv(x, fs, noverlap=0)
	    	SCVnoise[tr,:] = SCV
	elif color == 'colored':
			for tr in range(0,numtrs):
		    # colored noise
		    x = np.random.normal(size=(len(t)))
		    y = np.convolve(x,ker,mode='same')
		    if with_osc == True:
	    		if with_osc == True:
	    		f, SCV = spectral.scv(y+osc, fs, noverlap=0)
	    	else:
	    		f, SCV = spectral.scv(y, fs, noverlap=0)
	    	SCVnoise[tr,:] = SCV   

	return SCVnoise
