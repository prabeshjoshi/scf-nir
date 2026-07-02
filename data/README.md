# Datasets

The datasets analysed in the manuscript are third-party public benchmarks and
are **not redistributed** in this repository. Each is available from its
original source.

| Dataset            | n   | Source |
|--------------------|-----|--------|
| Wheat grain protein | 248 | 2016 IDRC Software Shoot-Out (USDA GIPSA); summarised in Igne et al., *NIR News* 2017, 28, 16–22 |
| Cocoa bean moisture | 72  | Mendeley Data, doi:10.17632/7734j4fd98.1 |
| Coffee–barley       | 158 | Ebrahimi-Najafabadi et al., *Talanta* 2012, 99, 175–179 |
| IDRC 2002 tablets   | 615×2 | IDRC 2002 shoot-out archive; Hopkins, *NIR News* 2003, 14, 10–13 |

## Expected format for the wheat examples

The wheat scripts read Excel spreadsheets with:

- column 0: sample ID (ignored)
- column 1: reference protein (%)
- columns 2+ : spectra, with **wavelength (nm) as the column header** of each

Place the four wheat files in a directory and pass it with `--data-dir`:

```
Cal_ManufacturerA.xlsx    Test_ManufacturerA.xlsx
Cal_ManufacturerB.xlsx    Test_ManufacturerB.xlsx
```

The Manufacturer B files are only needed for the cross-instrument transfer
section; the same-instrument result runs from the two Manufacturer A files
alone.

The code has no dependence on this particular file layout beyond
`examples/_wheat_data.py`; adapting the loader to another format is a two-line
change.
