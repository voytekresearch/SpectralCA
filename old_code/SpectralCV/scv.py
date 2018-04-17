"""
scv.py
functions needed to do analysis for coefficient of variation of power spectral density.

"""

import numpy as np

def scv(SP):
    """
    calculating the coefficient of variation

    Parameters
    ----------
    SP : array-like 1d
        Spectrogram of data.

    Returns
    -------
    scv : array-like 1d
        return the cofficient of variation by the spectrogram corresponds to the segment times.

    """
    scv = ((np.std(SP,axis=1)/np.mean(SP,axis=1)))
    return scv


def psd(SP):
    """
    calculating the power spectral density.

    Parameters
    ----------
    SP : array-like 1d
        Spectrogram of x.

    Returns
    -------
    psd : array-like 1d
        return the mean of each segment by the spectrogram.

    """
    psd = (np.mean(SP,axis=1))
    return psd
