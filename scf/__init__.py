"""
Spectral Contrast Features (SCF) — minimal reference implementation.

A bin-difference feature transformation for NIR calibration: predictive
features are constructed as differences between the mean intensities of paired
spectral bins, whose positions, widths, and sparsity are jointly optimised by a
genetic algorithm and passed to PLSR.

This package is the reference implementation accompanying the manuscript
"Spectral Contrast Features: A Bin-Difference Approach to Interpretable,
Parsimonious, and Cross-Instrument NIR Calibration". It is intentionally
minimal: it reproduces the method as described (Equations 1-2, Table 2,
Section 2.5) rather than providing a general-purpose toolkit.
"""

from .features import make_scf_features, chromosome_to_bins, n_variables_used
from .ga import run_ga, DEFAULTS
from .plsr import select_n_components, cv_rmse, cv_predict, fit_pls
from .evaluation import (
    regression_metrics,
    nested_cv_scf,
    cv_scf_fixed,
    cv_full_spectrum,
)
from . import preprocessing

__all__ = [
    "make_scf_features", "chromosome_to_bins", "n_variables_used",
    "run_ga", "DEFAULTS",
    "select_n_components", "cv_rmse", "cv_predict", "fit_pls",
    "regression_metrics", "nested_cv_scf", "cv_scf_fixed", "cv_full_spectrum",
    "preprocessing",
]

__version__ = "0.1.0"
