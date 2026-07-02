"""Synthetic self-test: verifies the SCF pipeline runs end to end.

Builds a spectrum-like matrix where the target is a genuine bin-difference of
two regions plus noise, then checks that (a) the GA finds a low-fitness
solution, (b) nested CV completes, and (c) a frozen chromosome reproduces
through the feature/metric path. Uses tiny GA settings for speed.
"""
import numpy as np
import sys
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scf import (make_scf_features, run_ga, nested_cv_scf,
                 cv_scf_fixed, cv_full_spectrum, regression_metrics,
                 chromosome_to_bins)

rng = np.random.default_rng(0)
n, p = 200, 120
wl = np.linspace(900, 1100, p)

# Smooth random spectra
base = np.cumsum(rng.normal(0, 1, size=(n, p)), axis=1)
base = (base - base.mean(0)) / base.std(0)
# True signal: difference of region A (idx 30-40) minus region B (idx 70-80)
true_feat = base[:, 30:40].mean(1) - base[:, 70:80].mean(1)
y = 12.0 + 3.5 * true_feat + rng.normal(0, 0.3, size=n)

print("=== GA (tiny settings) ===")
best, fit, hist = run_ga(
    base, y,
    pop_size=20, generations=15, P=6, max_width=20,
    n_components=None, cv=5, seed=0, verbose=True,
)
print("best fitness:", round(fit, 4), "| active:", int(best[:, 4].sum()))
print("selected bins (nm):")
for (b1, b2) in chromosome_to_bins(best, wl):
    print(f"  [{b1[0]:.1f}-{b1[1]:.1f}] - [{b2[0]:.1f}-{b2[1]:.1f}]")

print("\n=== Calibration CV on GA features ===")
m_scf, nc = cv_scf_fixed(base, y, best, seed=0)
print("SCF-PLSR:", {k: round(v, 3) for k, v in m_scf.items()}, "nc=", nc)
m_full, ncf = cv_full_spectrum(base, y, seed=0)
print("Full-spectrum:", {k: round(v, 3) for k, v in m_full.items()}, "nc=", ncf)

print("\n=== Nested CV (tiny GA) ===")
m_nested = nested_cv_scf(
    base, y, n_outer=3, seed=0,
    ga_kwargs=dict(pop_size=15, generations=8, P=6, n_components=None, verbose=False),
)
print("nested SCF:", {k: round(v, 3) for k, v in m_nested.items()})

# Sanity assertions
assert best[:, 4].sum() >= 1, "no active features"
assert m_scf["R2"] > 0.8, "SCF should recover the planted signal"
assert not np.isnan(m_nested["RMSE"]), "nested CV produced NaN"
print("\nALL CHECKS PASSED")
