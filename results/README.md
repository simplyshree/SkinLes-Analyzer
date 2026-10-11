# Benchmark outputs

`benchmark_results.csv` is written by `scripts/run_benchmark.py` or by the notebook. It contains one aggregate row per method with mean, standard deviation, median, and sample count for IoU and Dice, plus the source and deterministic evaluation split.

The file is only scientifically meaningful when `dataset` identifies an official ISIC evaluation, `evaluation_split` is `test`, and all methods were evaluated on the same held-out pairs. Local smoke runs are useful for regression testing but must not be presented as ISIC performance.

The benchmark command also writes:

- `benchmark_results_per_sample.csv` for per-image Dice/IoU and foreground counts.
- `split_manifest.csv` for deterministic train/validation/test membership.
- `run_metadata.json` for dataset, seed, counts, methods, and selected best method.
- `figures/comparison_grid.png` and `figures/metric_summary.png` for presentations.

Generated outputs may be regenerated. Dataset caches and model binaries remain outside Git.

