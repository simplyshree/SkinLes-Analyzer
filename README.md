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

## System workflow

```text
Official ISIC image + expert mask archives
                 │
                 ▼
       Cache, hash, extract, pair, validate
                 │
                 ▼
 Resize → RGB/LAB/HSV views → optional hair removal/CLAHE
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
     Otsu   K-means     U-Net
       └─────────┼─────────┘
                 ▼
 Morphological cleanup and binary mask validation
                 │
                 ▼
 Dice/IoU benchmark + comparison figures + reports
                 │
                 ▼
 Streamlit upload → live inference → mask/overlay/descriptors
```

## Image-processing theory

### Input and preprocessing

Every image is converted to RGB, checked for a valid shape, and resized to 256×256. Resizing gives every algorithm a consistent input while preserving the original upload for display. RGB is convenient for display, but segmentation can use grayscale or perceptual colour spaces.

- **RGB:** direct red, green, and blue intensities. Useful for display and colour statistics.
- **Grayscale:** a single luminance-like channel used by Otsu thresholding.
- **HSV:** separates hue, saturation, and value; useful when brightness and colourfulness should be reasoned about separately.
- **LAB:** separates lightness from opponent colour channels and is used by the K-means baseline because Euclidean colour distance is more meaningful than raw RGB distance for many images.

Optional preprocessing includes CLAHE, which improves local luminance contrast, and black-hat morphology plus inpainting for dark hair-like structures. These operations are not guaranteed to improve every image, so they are explicit UI options.

### Otsu thresholding

Otsu’s method searches for a grayscale threshold that best separates two intensity populations by maximising between-class variance. The binary result has no learned parameters, so it is fast and transparent. The implementation evaluates foreground polarity, retains a plausible connected component, and applies morphology to reduce isolated noise.

Otsu is sensitive to uneven illumination, low lesion/background contrast, rulers, hairs, and images whose histogram is not naturally bimodal.

### K-means segmentation

K-means assigns each pixel to one of `k` clusters by alternating between assigning points to the nearest centroid and updating each centroid. This project uses LAB pixels with a fixed seed and normally `k=2`. Candidate clusters are scored using simple geometry safeguards before morphological cleanup.

K-means does not understand lesion shape or anatomy. Its result can change when colour distributions overlap or when the background contains strong visual structure.

### Morphological cleanup

Morphology operates on binary masks with a structuring element:

- **Opening** removes small isolated foreground regions.
- **Closing** fills narrow gaps and joins nearby foreground regions.
- **Hole filling** repairs enclosed background regions.
- **Connected components** allow the pipeline to keep the most plausible lesion region.

These operations improve presentation quality but can also remove real small structures. They are therefore kept deterministic and tested independently.

### U-Net segmentation

U-Net is an encoder-decoder convolutional network. The encoder reduces spatial resolution while learning increasingly rich features. The decoder upsamples those features to recover pixel-level boundaries. Skip connections copy high-resolution encoder features into the decoder, helping preserve lesion edges.

The model predicts a probability for every pixel. Training uses a combined binary-cross-entropy and Dice-style objective:

```text
loss = binary_cross_entropy + (1 - dice)
```

Training is offline through `scripts/train_unet.py` or the notebook. The Streamlit app never starts training; it only loads an existing checkpoint and performs inference.

### Evaluation metrics

For binary masks:

```text
IoU   = intersection / union
Dice  = 2 × intersection / (predicted foreground + true foreground)
```

Both metrics are calculated per test image and then summarised with mean, standard deviation, and median. All methods in one benchmark use exactly the same held-out test pairs.

### ABCD-style image descriptors

The selected mask drives four educational descriptors:

- **A — Asymmetry:** disagreement between a lesion mask and mirrored versions around its bounding box.
- **B — Border:** area, perimeter, circularity, and a border-irregularity proxy.
- **C — Colour:** masked RGB mean, RGB standard deviation, colour variation, and compact dominant-colour clusters.
- **D — Diameter:** maximum axis-aligned extent in pixels.

These are image descriptors only. They are not validated clinical ABCD criteria. Diameter is not converted to millimetres because the images have no physical calibration metadata.

## Dataset

The default `official_isic` route uses the practical official Task 1 bundle containing 900 JPEG images and 900 binary masks. It is cached in `dataset_cache/`, verified with archive hashes, and checked for exact image/mask pairing. The larger ISIC 2018/KaggleHub route remains optional.

The dataset, archives, cache metadata, and trained model are ignored by Git. Do not commit patient images or large model binaries.

## Installation

PowerShell:

```powershell
cd "C:\Users\Sgoff\MYfile\Desktop\PYThh\Skin - Les Analyzer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow is needed for U-Net training or inference. Classical methods and most tests do not need to load TensorFlow.

## Reproducible experiments

The supported sizes are:

| Mode | Purpose |
|---|---|
| `20` | Smoke test and fast debugging |
| `100` | Quick validation |
| `500` | Practical development run |
| `all` | Main 900-pair experiment |

Train the U-Net offline:

```powershell
python scripts\train_unet.py --source official_isic --num-samples all --epochs 25 --model-path model\unet_skin_lesion.keras
```

Run the complete benchmark:

```powershell
python scripts\run_benchmark.py `
  --source official_isic `
  --num-samples all `
  --model-path model\unet_skin_lesion.keras
```

The benchmark writes:

- `results/benchmark_results.csv` — aggregate method metrics.
- `results/benchmark_results_per_sample.csv` — one row per method and test image.
- `results/split_manifest.csv` — deterministic train/validation/test membership.
- `results/run_metadata.json` — seed, counts, methods, paths, and selected best method.
- `results/figures/comparison_grid.png` — representative original/ground-truth/prediction panels.
- `results/figures/metric_summary.png` — Dice and IoU comparison chart.

For a faster run, replace `all` with `20`, `100`, or `500`.

## Run the Streamlit frontend

```powershell
streamlit run app.py
```

The frontend has three tabs:

1. **Analyze image:** upload an image, choose a method, view the original/preprocessed/mask/overlay panels, and download PNG/JSON artifacts.
2. **Benchmark results:** inspect the executed method table, ranking, split counts, and charts.
3. **How it works:** explain the processing theory and limitations during a demo or presentation.

The application reads measured benchmark results. If a U-Net method is selected but its checkpoint is unavailable, the app reports the problem instead of pretending a neural prediction exists.

## Tests and quality checks

```powershell
python -m pytest -q
python -m compileall -q analyzer scripts app.py
```

The test suite covers pairing, deterministic splitting, preprocessing, segmentation, metrics, pipeline behaviour, model validation, and edge cases. A headless Streamlit startup should also be checked before a presentation or deployment.

## Presentation outline

Use this sequence for a five-minute explanation:

1. State the problem: convert a dermoscopic image into a lesion mask.
2. Explain the common preprocessing representation and why segmentation is needed before measurements.
3. Compare a transparent baseline (Otsu), an unsupervised colour baseline (K-means), and a learned model (U-Net).
4. Show that all methods are evaluated on the same deterministic test set using Dice and IoU.
5. Demonstrate the Streamlit upload flow and the four visual outputs.
6. Explain that A/B/C/D values are descriptive image features, not a medical assessment.
7. Discuss failure cases, dataset size, and why the measured best method is selected rather than assumed.

## Documentation map

- `ARCHITECTURE.md` — implementation and data-flow design.
- `PRODUCT.md` — users, goals, workflows, and scope.
- `AGENT.md` — development and handoff rules.
- `ADR-001.md` — key architecture and dataset decisions.
- `VALIDATION.md` — evidence-backed validation status.

## Limitations

Segmentation quality can degrade with unusual illumination, occlusion, hair, rulers, low contrast, border-touching lesions, or images outside the training distribution. The official small dataset is useful for a reproducible educational project but is not a universal clinical validation set. No public deployment should be described as clinically validated.
