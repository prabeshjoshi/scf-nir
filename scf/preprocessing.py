"""
Spectral preprocessing (Section 2.2).

Only the transforms used in the manuscript are provided: wavelength-window
truncation and Savitzky-Golay derivatives (window 7, polynomial order 2), and
SNV. Preprocessing is applied uniformly across a dataset with the same
parameters for calibration and test partitions.
"""

import numpy as np
from scipy.signal import savgol_filter


def truncate(X, wl, wl_lo, wl_hi):
    """Restrict to an open wavelength window (wl_lo, wl_hi)."""
    wl = np.asarray(wl, dtype=float)
    mask = (wl > wl_lo) & (wl < wl_hi)
    return np.asarray(X, dtype=float)[:, mask], wl[mask]


def savgol(X, window=7, polyorder=2, deriv=0):
    """Savitzky-Golay filter / derivative along the wavelength axis."""
    return savgol_filter(np.asarray(X, dtype=float),
                         window_length=window, polyorder=polyorder, deriv=deriv)


def snv(X):
    """Standard Normal Variate: centre and scale each spectrum individually."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1e-12
    return (X - mu) / sd
