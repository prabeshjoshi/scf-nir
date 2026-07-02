"""
Spectral Contrast Feature (SCF) construction.

Implements Equation (1) of the manuscript:

    f_k = mean( X[:, s1 : s1 + w1] ) - mean( X[:, s2 : s2 + w2] )

Each contrast feature is the difference between the mean intensities of two
spectral bins. A candidate solution (chromosome) is a set of P feature pairs,
each encoded as a 5-element gene [s1, w1, s2, w2, active]; only genes with
active == 1 contribute to the feature matrix (Section 2.3).
"""

import numpy as np


def make_scf_features(X, chromosome):
    """Construct the SCF feature matrix from preprocessed spectra.

    Parameters
    ----------
    X : ndarray (n_samples, n_variables)
        Preprocessed spectral matrix.
    chromosome : ndarray (P, 5), integer
        Rows are genes [s1, w1, s2, w2, active].

    Returns
    -------
    F : ndarray (n_samples, n_active)
        One column per active contrast feature. Empty (n_samples, 0) if no
        gene is active.
    active_idx : list[int]
        Row indices of the active genes, in column order of F.
    """
    X = np.asarray(X, dtype=float)
    chromosome = np.asarray(chromosome)
    n_samples, n_var = X.shape

    cols, active_idx = [], []
    for i, gene in enumerate(chromosome):
        s1, w1, s2, w2, active = (int(v) for v in gene)
        if not active:
            continue
        # Clip bins to valid ranges so an out-of-range gene never crashes;
        # width is at least 1 variable.
        s1 = max(0, min(s1, n_var - 1))
        w1 = max(1, min(w1, n_var - s1))
        s2 = max(0, min(s2, n_var - 1))
        w2 = max(1, min(w2, n_var - s2))
        mean1 = X[:, s1:s1 + w1].mean(axis=1)
        mean2 = X[:, s2:s2 + w2].mean(axis=1)
        cols.append((mean1 - mean2).reshape(-1, 1))
        active_idx.append(i)

    if not cols:
        return np.zeros((n_samples, 0)), active_idx
    return np.hstack(cols), active_idx


def chromosome_to_bins(chromosome, wl):
    """Return, for each active gene, the two (start_nm, end_nm) bin edges.

    Useful for the chemical interpretation of selected features (Section 3.x).
    """
    chromosome = np.asarray(chromosome)
    wl = np.asarray(wl, dtype=float)
    n_var = len(wl)
    out = []
    for gene in chromosome:
        s1, w1, s2, w2, active = (int(v) for v in gene)
        if not active:
            continue
        s1 = max(0, min(s1, n_var - 1)); w1 = max(1, min(w1, n_var - s1))
        s2 = max(0, min(s2, n_var - 1)); w2 = max(1, min(w2, n_var - s2))
        bin1 = (wl[s1], wl[s1 + w1 - 1])
        bin2 = (wl[s2], wl[s2 + w2 - 1])
        out.append((bin1, bin2))
    return out


def n_variables_used(chromosome, n_var):
    """Number of distinct spectral variables covered by all active bins."""
    chromosome = np.asarray(chromosome)
    used = set()
    for gene in chromosome:
        s1, w1, s2, w2, active = (int(v) for v in gene)
        if not active:
            continue
        used.update(range(max(0, s1), min(n_var, s1 + w1)))
        used.update(range(max(0, s2), min(n_var, s2 + w2)))
    return len(used)
