"""
PLSR fitting utilities.

The downstream regressor is PLSR (Section 2.5). Two behaviours from the
manuscript are implemented here:

* Latent-variable selection by an inner cross-validation loop, "up to 10
  latent variables" (Section 2.4). This is used for both the SCF feature
  matrix and the full-spectrum baseline, so the baseline is never fixed at an
  arbitrary component count.
* A cross-validated RMSE used as the GA fitness metric (Equation 2).

Near-constant columns are dropped inside each fold and the component count is
capped at min(max_lv, n_features, n_train - 1) so degenerate chromosomes never
raise.
"""

import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

MAX_LV = 10  # "up to 10 latent variables" (Section 2.4)


def _fit_predict(Xtr, ytr, Xte, n_components):
    """Fit PLSR on a fold, dropping near-constant training columns."""
    stds = Xtr.std(axis=0)
    keep = stds > 1e-8
    if not np.any(keep):
        return None
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    n_comp = min(n_components, Xtr.shape[1], Xtr.shape[0] - 1)
    n_comp = max(1, n_comp)
    pls = PLSRegression(n_components=n_comp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pls.fit(Xtr, ytr)
        return pls.predict(Xte).ravel()


def select_n_components(X, y, max_lv=MAX_LV, cv=5, seed=0):
    """Choose the PLSR latent-variable count by inner cross-validation.

    Returns the component count in 1..cap that minimises mean fold RMSE, where
    cap = min(max_lv, n_features, n_samples - 1).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    cap = min(max_lv, X.shape[1], X.shape[0] - 1)
    if cap < 1:
        return 1
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    best_nc, best_rmse = 1, np.inf
    for nc in range(1, cap + 1):
        errs = []
        for tr, te in kf.split(X):
            yp = _fit_predict(X[tr], y[tr], X[te], nc)
            if yp is None:
                errs = None
                break
            errs.append(np.sqrt(mean_squared_error(y[te], yp)))
        if errs is None:
            continue
        m = float(np.mean(errs))
        if m < best_rmse:
            best_rmse, best_nc = m, nc
    return best_nc


def cv_rmse(X, y, n_components=None, cv=5, seed=0, max_lv=MAX_LV):
    """Cross-validated RMSE used as the GA fitness term (Equation 2).

    If ``n_components`` is None, the component count is chosen by inner
    cross-validation within each outer fold (paper-faithful). Passing an
    integer fixes the component count instead (faster, for quick runs).
    Returns np.inf for an empty feature matrix so the GA rejects it.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.shape[1] == 0:
        return np.inf
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    errs = []
    for tr, te in kf.split(X):
        if n_components is None:
            nc = select_n_components(X[tr], y[tr], max_lv=max_lv, cv=cv, seed=seed)
        else:
            nc = n_components
        yp = _fit_predict(X[tr], y[tr], X[te], nc)
        if yp is None:
            return np.inf
        errs.append(np.sqrt(mean_squared_error(y[te], yp)))
    return float(np.mean(errs))


def cv_predict(X, y, n_components, cv=5, seed=0):
    """Out-of-fold predictions for reporting calibration CV metrics."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    yp = np.zeros_like(y, dtype=float)
    for tr, te in kf.split(X):
        pred = _fit_predict(X[tr], y[tr], X[te], n_components)
        yp[te] = np.nan if pred is None else pred
    return yp


def fit_pls(X, y, n_components):
    """Fit a final PLSR model on all training data with a fixed nc."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_comp = max(1, min(n_components, X.shape[1], X.shape[0] - 1))
    pls = PLSRegression(n_components=n_comp)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pls.fit(X, y)
    return pls
