"""
Wheat protein — end-to-end SCF-PLSR (rediscovers the contrast features).

Runs the genetic algorithm on the wheat calibration set with the Table 2
hyperparameters, reports the selected contrast features and their chemical
regions, and evaluates SCF-PLSR against the full-spectrum PLSR baseline with a
nested cross-validation (Section 2.5).

The search is stochastic: the exact bins and metrics will vary slightly between
runs and seeds. This does not affect the conclusions, and the recovered regions
consistently fall on the N-H / O-H second-overtone bands (Section 3.1). For an
exact, deterministic reproduction of the reported numbers, use
``wheat_reproduce.py``.

Usage:
    python examples/wheat_scf_example.py --data-dir /path/to/wheat
    python examples/wheat_scf_example.py --data-dir /path/to/wheat --quick
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scf import (run_ga, make_scf_features, chromosome_to_bins,
                 n_variables_used, cv_scf_fixed, cv_full_spectrum,
                 nested_cv_scf, DEFAULTS)
from scf.preprocessing import truncate, savgol
from examples._wheat_data import load_dataset


def preprocess(X, wl):
    Xt, wlt = truncate(X, wl, 780, 1100)
    return savgol(Xt, window=7, polyorder=2, deriv=2), wlt


def main(data_dir, quick, seed):
    X_raw, y, wl_full = load_dataset(os.path.join(data_dir, "Cal_ManufacturerA.xlsx"))
    X, wl = preprocess(X_raw, wl_full)
    print(f"Wheat calibration: {X.shape[0]} samples, {X.shape[1]} variables, "
          f"{wl.min():.1f}-{wl.max():.1f} nm")

    # Table 2 settings by default; --quick shrinks the search for a fast demo.
    if quick:
        ga_kwargs = dict(pop_size=30, generations=40, n_components=None, verbose=True)
        print("Mode: quick demonstration (reduced GA budget)\n")
    else:
        ga_kwargs = dict(
            pop_size=DEFAULTS["pop_size"], generations=DEFAULTS["generations"],
            P=DEFAULTS["P"], max_width=DEFAULTS["max_width"],
            cx_prob=DEFAULTS["cx_prob"], mut_prob=DEFAULTS["mut_prob"],
            tournament_k=DEFAULTS["tournament_k"], lam=DEFAULTS["lam"],
            cv=DEFAULTS["cv"], n_components=None, verbose=True,
        )
        print("Mode: full Table 2 settings (this takes a few minutes)\n")

    print("Running genetic algorithm on the full calibration set ...")
    best, fit, _ = run_ga(X, y, seed=seed, **ga_kwargs)
    n_active = int(best[:, 4].sum())
    print(f"\nSelected {n_active} contrast features "
          f"({n_variables_used(best, X.shape[1])} of {X.shape[1]} variables); "
          f"fitness={fit:.4f}")
    for i, (b1, b2) in enumerate(chromosome_to_bins(best, wl), 1):
        print(f"  F{i}: [{b1[0]:.1f}-{b1[1]:.1f}] - [{b2[0]:.1f}-{b2[1]:.1f}] nm")

    print("\nCalibration cross-validation:")
    m_scf, nc = cv_scf_fixed(X, y, best, seed=seed)
    print(f"  SCF-PLSR (nc={nc}) :", {k: round(v, 3) for k, v in m_scf.items()})
    m_fs, ncf = cv_full_spectrum(X, y, seed=seed)
    print(f"  Full-spectrum (nc={ncf}):", {k: round(v, 3) for k, v in m_fs.items()})

    print("\nNested cross-validation (GA re-run inside each outer fold):")
    nested_kwargs = dict(ga_kwargs)
    nested_kwargs["verbose"] = False
    if not quick:
        # Keep the nested loop tractable at full settings.
        nested_kwargs.update(generations=120)
    m_nested = nested_cv_scf(X, y, n_outer=5, seed=seed, ga_kwargs=nested_kwargs)
    print("  SCF-PLSR nested:", {k: round(v, 3) for k, v in m_nested.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="End-to-end wheat SCF-PLSR.")
    ap.add_argument("--data-dir", default=".",
                    help="Directory containing Cal_ManufacturerA.xlsx.")
    ap.add_argument("--quick", action="store_true",
                    help="Reduced GA budget for a fast demonstration.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.data_dir, a.quick, a.seed)
