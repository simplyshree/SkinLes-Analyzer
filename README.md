# Skin Lesion Analyzer

An educational image-processing system for segmenting dermoscopic skin-lesion images and extracting mask-based visual descriptors. The project compares classical computer-vision methods with a compact U-Net and exposes the result through a presentation-ready Streamlit application.

> This project is not a medical device. It does not diagnose skin disease, estimate cancer risk, or replace clinical review.

## What the project demonstrates

- Image standardisation, colour-space conversion, contrast enhancement, and optional hair-artifact removal.
- Classical segmentation with Otsu thresholding and K-means clustering.
- Morphological opening, closing, hole filling, polarity selection, and connected-component filtering.
- A compact U-Net trained offline on expert masks.
- Reproducible train/validation/test splitting and held-out evaluation.
- Dice and IoU comparison with mean, standard deviation, median, and per-sample metrics.
- ABCD-style image descriptors derived from the predicted mask.
- A Streamlit frontend for interactive inference and presentation.

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
