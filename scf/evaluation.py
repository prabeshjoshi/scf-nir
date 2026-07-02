"""
Performance metrics and the nested cross-validation evaluation of Section 2.5.

The nested loop runs the GA inside each outer training partition, so the
spectral regions chosen by the GA in a given fold are never used to compute
that fold's performance estimate.
"""

import numpy as np
from sklearn.model_selection import KFold

from .features import make_scf_features
from .plsr import select_n_components, cv_predict, fit_pls
from .ga import run_ga


def regression_metrics(y_true, y_pred):
    """R^2, RMSE, RPD, and bias for a set of point predictions."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    rpd = float(y_true.std(ddof=1) / rmse) if rmse > 0 else float("nan")
    bias = float(resid.mean())
    return {"R2": r2, "RMSE": rmse, "RPD": rpd, "Bias": bias}


def nested_cv_scf(X, y, n_outer=5, seed=0, ga_kwargs=None):
    """Nested CV for SCF-PLSR (Section 2.5).

    For each of ``n_outer`` outer folds: run the GA on the outer-training
    partition (its fitness uses an independent inner CV), fit PLSR on that
    partition with the GA-selected features, and predict the held-out outer
    fold. Returns pooled out-of-fold metrics.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ga_kwargs = dict(ga_kwargs or {})
    kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)

    y_oof = np.full_like(y, np.nan, dtype=float)
    for k, (tr, te) in enumerate(kf.split(X)):
        best_chrom, _, _ = run_ga(X[tr], y[tr], seed=seed + k, **ga_kwargs)
        F_tr, _ = make_scf_features(X[tr], best_chrom)
        F_te, _ = make_scf_features(X[te], best_chrom)
        nc = select_n_components(F_tr, y[tr], seed=seed + k)
        model = fit_pls(F_tr, y[tr], nc)
        y_oof[te] = model.predict(F_te).ravel()
    return regression_metrics(y, y_oof)


def cv_scf_fixed(X, y, chromosome, cv=5, seed=0):
    """Calibration CV metrics for a fixed (already-selected) chromosome.

    Used to report headline calibration numbers once the GA has produced a
    feature set on the full training data, and by the reproduction script.
    """
    F, _ = make_scf_features(X, chromosome)
    nc = select_n_components(F, y, seed=seed)
    y_cv = cv_predict(F, y, nc, cv=cv, seed=seed)
    return regression_metrics(y, y_cv), nc


def cv_full_spectrum(X, y, cv=5, seed=0):
    """Full-spectrum PLSR baseline with inner-CV component selection.

    The component count is chosen by inner CV rather than fixed, so the
    baseline is not inadvertently inflated or deflated.
    """
    nc = select_n_components(X, y, seed=seed)
    y_cv = cv_predict(X, y, nc, cv=cv, seed=seed)
    return regression_metrics(y, y_cv), nc
