# Results

Every training run writes its outputs here. On Colab the notebooks point
`output_dir` at `<project_root>/results/` on Drive so artifacts persist
between sessions.

```
results/
  logs/<run_name>.csv             per-epoch train/val loss + epoch time
  metrics/<run_name>.json         final test metrics + the run config that produced them
  metrics/<run_name>.csv          one-row mirror of the JSON for easy aggregation
  checkpoints/<run_name>.pt       best-by-val-MSE (gitignored — too large for the repo)
  figures/<run_name>_training_curve.png
  figures/<run_name>_prediction_plot.png
  figures/<aggregate>.png         poster figures produced by notebook 05
  tables/                         aggregated tables (main_results.csv etc.)
```

The aggregate poster figures live alongside the per-run plots in
`figures/`. The README files in `poster/` and `report/` reference these
paths directly.

Checkpoints and `.log` files are gitignored — they are large and
regenerable from the configs. Figures, metrics, and tables are tracked so
the deliverable is self-contained.

## Headline aggregate figures (referenced by README + poster)

- `figures/main_results_bar.png`, `figures/main_results_heatmap.png`
- `figures/patching_scaling.png`, `figures/patching_efficiency.png`
- `figures/channel_independence_vs_mixing.png`
- `figures/patch_size_sweep.png`, `figures/lookback_sweep.png`
- `figures/seed_robustness.png`, `figures/pareto_params_vs_mse.png`
- `figures/ssl_vs_cold.png`, `figures/traffic_results.png`
- `figures/example_predictions.png`, `figures/failure_case_*.png`

All produced by `notebooks/05_generate_results_figures_colab.ipynb` from
the per-run JSONs in `metrics/`.
