# Datasets

The benchmark CSVs are **not** stored in this repo — they're large and
redistribution is restricted. Place them under
`<project_root>/data/<name>/<name>.csv` (or `<project_root>/data/<name>.csv`).

When running on Colab, `<project_root>` is
`/content/drive/MyDrive/cs4782_patchtst_project`, so the canonical paths are:

```
/content/drive/MyDrive/cs4782_patchtst_project/data/weather/weather.csv
/content/drive/MyDrive/cs4782_patchtst_project/data/electricity/electricity.csv
/content/drive/MyDrive/cs4782_patchtst_project/data/traffic/traffic.csv   # optional
```

## Where to download

The PatchTST authors and the related Autoformer/Informer projects host the
standard long-term forecasting benchmark CSVs:

- Autoformer release: <https://github.com/thuml/Autoformer> (see *Datasets* in
  the README — it links a Google Drive folder containing `weather.csv`,
  `electricity.csv`, `traffic.csv`, and the ETT files).
- PatchTST repo: <https://github.com/yuqinie98/PatchTST> (uses the same files).

## Expected file format

Each CSV has a leading `date` column (parsed and discarded) followed by
numerical feature columns:

```
date,T (degC),OT,...,WV (m/s)
2020-01-01 00:00:00,2.7,...
```

Sizes (approximate):

| dataset      | rows   | features | sampling |
|--------------|--------|----------|----------|
| Weather      | ~52.7k | 21       | 10 min   |
| Electricity  | ~26.3k | 321      | 1 hour   |
| Traffic      | ~17.5k | 862      | 1 hour   |

## Verification

`notebooks/00_colab_setup_and_data.ipynb` loads each CSV, prints its shape,
then constructs one sliding-window batch and prints the tensor shapes. Run
that first.
