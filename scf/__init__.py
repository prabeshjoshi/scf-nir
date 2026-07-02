"""
Spectral Contrast Features (SCF): reference implementation.

A bin-difference feature transformation for NIR calibration: predictive
features are constructed as differences between the mean intensities of paired
spectral bins, whose positions, widths, and sparsity are jointly optimised by a
genetic algorithm and passed to PLSR.

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
