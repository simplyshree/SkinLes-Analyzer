# Benchmark outputs

`benchmark_results.csv` is written by `scripts/run_benchmark.py` or by the Colab notebook. It contains one aggregate row per method with mean, standard deviation, median, and sample count for IoU and Dice.

The file is only scientifically meaningful when `dataset` identifies an ISIC 2018 evaluation and all methods were evaluated on the same held-out pairs. Local smoke runs are useful for regression testing but must not be presented as ISIC performance.

