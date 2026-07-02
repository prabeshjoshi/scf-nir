# Spectral Contrast Features (SCF) — reference implementation

Reference implementation accompanying:

> **Spectral Contrast Features: A Bin-Difference Approach to Interpretable,
> Parsimonious, and Cross-Instrument NIR Calibration.** Prabesh Joshi.

SCF constructs predictive features as **differences between the mean
intensities of paired spectral bins**, whose positions, widths, and sparsity
are jointly optimised by a genetic algorithm and passed to PLSR.

This repository is deliberately **minimal**: it implements the method exactly as
described in the manuscript so that the reported results can be verified, rather
than providing a general-purpose toolkit. A broader package with additional
feature forms and utilities is maintained separately.

## What is here

| File | Manuscript reference |
|------|----------------------|
| `scf/features.py` | Contrast-feature construction — **Equation (1)**; chromosome encoding `[s1, w1, s2, w2, active]` (Section 2.3) |
| `scf/ga.py` | Genetic-algorithm optimisation of the penalised fitness — **Equation (2)**; default hyperparameters reproduce **Table 2** (Section 2.4) |
| `scf/plsr.py` | PLSR with latent-variable selection by inner cross-validation, "up to 10 latent variables" (Section 2.4); full-spectrum baseline uses the same selection so it is never fixed at an arbitrary component count |
| `scf/evaluation.py` | Nested cross-validation and metrics — **Section 2.5** |
| `scf/preprocessing.py` | Wavelength truncation, Savitzky–Golay derivatives, SNV — **Section 2.2** |
| `examples/wheat_reproduce.py` | Deterministic reproduction of the wheat SG² result (**Table 3**) and cross-instrument transfer (**Section 3.1.5**) from the GA-selected feature set |
| `examples/wheat_scf_example.py` | End-to-end run that **rediscovers** the contrast features with the genetic algorithm |
| `selftest.py` | Runs the whole pipeline on synthetic data — no dataset required |

## Method in one page

A contrast feature is

```
f_k = mean( X[:, s1 : s1+w1] )  -  mean( X[:, s2 : s2+w2] )
```

A candidate solution (chromosome) holds up to `P` such feature pairs; a binary
`active` gene per pair lets the search choose how many features to use. The GA
minimises

```
fitness = RMSE_cv( X_train, y | active SCF features )  +  lambda * n_active
```

by tournament selection, feature-level crossover, and per-gene mutation, with
elitism. Performance is estimated by a nested cross-validation in which the GA
is re-run inside each outer training partition, so the regions selected in a
fold never inform that fold's error estimate.

## Install and run

```bash
pip install -r requirements.txt

# No data needed — verifies the implementation runs:
python selftest.py

# Reproduce the wheat headline result (seconds; uses the GA-selected features):
python examples/wheat_reproduce.py --data-dir /path/to/wheat

# Rediscover the features from scratch with the genetic algorithm:
python examples/wheat_scf_example.py --data-dir /path/to/wheat          # Table 2 settings
python examples/wheat_scf_example.py --data-dir /path/to/wheat --quick  # fast demo
```

Datasets are public third-party benchmarks and are not redistributed here; see
[`data/README.md`](data/README.md) for sources and the expected file format.

## Reproducibility note

The genetic algorithm is stochastic. The exact bins and metrics vary slightly
between runs and random seeds; this is expected and does not affect the
method's conclusions. `wheat_reproduce.py` uses the exact feature set selected
on the calibration data and is deterministic — its three features decode to the
wavelength regions reported in Section 3.1.2:

```
F1: [1010.0-1019.5] - [1000.5-1008.0] nm   (N-H second overtone)
F2: [978.5-988.0]   - [898.5-908.0]   nm   (water O-H second overtone)
F3: [855.0-862.0]   - [1012.0-1021.5] nm
```

## Suggested code-availability statement

> A reference implementation reproducing the results reported in this
> manuscript is available at <REPO-URL> (archived at <ZENODO-DOI>). The
> datasets analysed are public third-party benchmarks available from their
> original sources.

## License

MIT — see [`LICENSE`](LICENSE).
