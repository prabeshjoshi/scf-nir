"""Data loading for the wheat examples.

Expects the 2016 IDRC wheat protein files as Excel spreadsheets with:
    column 0 : sample ID (ignored)
    column 1 : reference protein (%)
    columns 2+ : spectra, with wavelength (nm) as the column headers

See data/README.md for how to obtain the dataset. The files are not
redistributed here.
"""

import numpy as np
import pandas as pd


def load_dataset(path):
    df = pd.read_excel(path)
    y = np.asarray(df.iloc[:, 1], dtype=float)
    X = np.asarray(df.iloc[:, 2:], dtype=float)
    wl = np.asarray([float(c) for c in df.columns[2:]], dtype=float)
    return X, y, wl
