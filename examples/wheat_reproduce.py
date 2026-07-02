"""
Wheat protein — reproduction of the reported SCF-PLSR result (Table 3, SG2).

This script uses the exact three-feature chromosome selected by the genetic
algorithm on the wheat calibration set, so it reproduces the headline numbers
deterministically and in seconds, without re-running the stochastic search.
It mirrors the same-instrument result in Table 3 and the cross-instrument
transfer of Section 3.1.5.

To rediscover the features from scratch, run ``wheat_scf_example.py`` instead.

Usage:
    python examples/wheat_reproduce.py --data-dir /path/to/wheat
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scf import (make_scf_features, chromosome_to_bins, cv_scf_fixed,
                 cv_full_spectrum, regression_metrics, fit_pls,
                 select_n_components)
from scf.preprocessing import truncate, savgol
from examples._wheat_data import load_dataset

# GA-selected three-feature solution on the wheat calibration set
# (indices are on the 780-1100 nm, SG2-preprocessed axis).
BEST_CHROMOSOME = np.array([
    [459, 20, 440, 16, 1],   # F1
    [396, 20, 236, 20, 1],   # F2
    [149, 15, 463, 20, 1],   # F3
], dtype=int)


def preprocess(X, wl):
    Xt, wlt = truncate(X, wl, 780, 1100)
    return savgol(Xt, window=7, polyorder=2, deriv=2), wlt


def main(data_dir):
    f = lambda name: os.path.join(data_dir, name)

    X_cal_raw, y_cal, wl_full = load_dataset(f("Cal_ManufacturerA.xlsx"))
    X_test_raw, y_test, _ = load_dataset(f("Test_ManufacturerA.xlsx"))
    X_cal, wl = preprocess(X_cal_raw, wl_full)
    X_test, _ = preprocess(X_test_raw, wl_full)
    print(f"Calibration {X_cal.shape}, test {X_test.shape}, "
          f"window {wl.min():.1f}-{wl.max():.1f} nm\n")

    print("Selected contrast features (nm):")
    for i, (b1, b2) in enumerate(chromosome_to_bins(BEST_CHROMOSOME, wl), 1):
        print(f"  F{i}: [{b1[0]:.1f}-{b1[1]:.1f}] - [{b2[0]:.1f}-{b2[1]:.1f}]")

    # ---- SCF-PLSR ----
    F_cal, _ = make_scf_features(X_cal, BEST_CHROMOSOME)
    F_test, _ = make_scf_features(X_test, BEST_CHROMOSOME)
    m_cal, nc = cv_scf_fixed(X_cal, y_cal, BEST_CHROMOSOME, seed=0)
    scf_model = fit_pls(F_cal, y_cal, nc)
    m_test = regression_metrics(y_test, scf_model.predict(F_test).ravel())
    print("\nSCF-PLSR (3 features)")
    print("  calibration CV:", {k: round(v, 3) for k, v in m_cal.items()})
    print("  held-out test :", {k: round(v, 3) for k, v in m_test.items()})

    # ---- Full-spectrum baseline (inner-CV component selection) ----
    m_fs_cal, ncf = cv_full_spectrum(X_cal, y_cal, seed=0)
    fs_model = fit_pls(X_cal, y_cal, ncf)
    m_fs_test = regression_metrics(y_test, fs_model.predict(X_test).ravel())
    print(f"\nFull-spectrum PLSR ({X_cal.shape[1]} variables, nc={ncf})")
    print("  calibration CV:", {k: round(v, 3) for k, v in m_fs_cal.items()})
    print("  held-out test :", {k: round(v, 3) for k, v in m_fs_test.items()})

    # ---- Cross-instrument transfer to Manufacturer B (Section 3.1.5) ----
    b_cal = f("Cal_ManufacturerB.xlsx")
    b_test = f("Test_ManufacturerB.xlsx")
    if os.path.exists(b_cal) and os.path.exists(b_test):
        XB_cal_raw, yB_cal, wlB = load_dataset(b_cal)
        XB_test_raw, yB_test, _ = load_dataset(b_test)
        XB_cal, _ = preprocess(XB_cal_raw, wlB)
        XB_test, _ = preprocess(XB_test_raw, wlB)
        XB = np.vstack([XB_cal, XB_test])
        yB = np.concatenate([yB_cal, yB_test])
        FB, _ = make_scf_features(XB, BEST_CHROMOSOME)

        print("\nDirect transfer to Manufacturer B (no recalibration)")
        m_fs_B = regression_metrics(yB, fs_model.predict(XB).ravel())
        m_scf_B = regression_metrics(yB, scf_model.predict(FB).ravel())
        print("  full-spectrum:", {k: round(v, 3) for k, v in m_fs_B.items()})
        print("  SCF-PLSR    :", {k: round(v, 3) for k, v in m_scf_B.items()})

        print("\nPer-feature univariate correlations (preserved across instruments)")
        for i in range(F_cal.shape[1]):
            r_cal = np.corrcoef(F_cal[:, i], y_cal)[0, 1]
            r_test = np.corrcoef(F_test[:, i], y_test)[0, 1]
            r_B = np.corrcoef(FB[:, i], yB)[0, 1]
            print(f"  F{i+1}: r(A-cal)={r_cal:+.3f}  "
                  f"r(A-test)={r_test:+.3f}  r(B)={r_B:+.3f}")
    else:
        print("\n(Manufacturer B files not found; skipping transfer section.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reproduce wheat SCF-PLSR result.")
    ap.add_argument("--data-dir", default=".",
                    help="Directory containing the wheat Excel files.")
    main(ap.parse_args().data_dir)
