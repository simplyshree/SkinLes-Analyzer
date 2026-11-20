# Skin Lesion Analyzer

Image-only educational project for lesion segmentation and descriptive ABCD-style image features.

## Run the local frontend

1. Install Python 3.11 or newer from https://www.python.org/downloads/.
2. Double-click `run_app.bat`.
3. Open http://localhost:8501.

The launcher creates `.venv`, installs the lightweight frontend dependencies,
and starts Streamlit. It uses the trained checkpoint at
`model/unet_skin_lesion.keras` when available and otherwise uses the classical
segmentation methods.

## Project contents

```text
app.py                              Streamlit frontend
analyzer/                           Image-processing and model code
model/unet_skin_lesion.keras        Trained U-Net checkpoint
notebooks/skin_lesion_experiments.ipynb  Colab experiment notebook
results/benchmark_results.csv       Measured benchmark table
results/run_metadata.json           Benchmark split metadata
requirements-app.txt                Frontend-only dependencies
run_app.bat                         Windows one-click launcher
run_app.ps1                         Launcher implementation
```

## Scope

The application accepts an image, segments the lesion, and reports descriptive
shape and colour features. It is not a medical diagnosis or clinical decision tool.
