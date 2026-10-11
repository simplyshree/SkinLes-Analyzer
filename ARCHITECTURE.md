# Architecture

## System boundary

Skin Lesion Analyzer is a local Python application with three boundaries:

1. **Data and experiments:** download/cache/verify the official image-mask pairs, split them deterministically, train offline, and benchmark on held-out data.
2. **Analysis library:** normalize images, run segmentation, calculate metrics and mask-derived descriptors, and return structured results.
3. **Presentation frontend:** Streamlit loads the benchmark/checkpoint and performs live inference for an uploaded image.

The frontend does not download datasets, train models, or silently create benchmark values during an upload.

## Components

| Component | Responsibility |
|---|---|
| `analyzer/data_loader.py` | Official/Kaggle/KaggleHub sources, archive cache, hashing, extraction safety, pairing, deterministic splitting |
| `analyzer/preprocessing.py` | RGB validation, resize, grayscale/HSV/LAB views, CLAHE, hair-artifact removal |
| `analyzer/classical_segmentation.py` | Otsu, K-means, polarity selection, morphology, connected-component cleanup |
| `analyzer/unet_segmentation.py` | U-Net architecture, loss, checkpoint loading, inference validation |
| `analyzer/metrics.py` | Binary Dice, IoU, and aggregate summaries |
| `analyzer/abcd_features.py` | Descriptive geometry and colour statistics from a binary mask |
| `analyzer/pipeline.py` | One analysis contract used by the app and notebook |
| `analyzer/reporting.py` | Pipeline version, JSON-safe reports, image downloads |
| `scripts/train_unet.py` | Offline training and checkpoint/test reporting |
| `scripts/run_benchmark.py` | Split manifest, aggregate/per-sample metrics, figures, metadata |
| `app.py` | Streamlit inference, benchmark display, theory, downloads |

## Data flow

```text
Archive URL/Kaggle source
        ↓
Cache archive + SHA-256 metadata
        ↓
Safe extraction + normalized filename index
        ↓
Exact image/mask pairing
        ↓
Seeded train/validation/test split
        ├── train/validation → offline U-Net training
        └── test → Otsu/K-means/U-Net comparison
                         ↓
                 CSV + JSON + figures
                         ↓
                   Streamlit reads artifacts
```

The split manifest is the reproducibility anchor. All methods receive the same test pair list and the same 256×256 evaluation representation.

## Result contracts

`analyze_lesion()` returns:

- `original`, `preprocessed`, `mask`, `contour_overlay`, and `mask_overlay` arrays;
- `selected_method`, `warning`, and ordered `processing_steps`;
- `abcd` measurements;
- `metadata` containing pipeline version, shapes, selected/requested method, preprocessing options, benchmark path, and model path.

The benchmark writes aggregate rows with `dataset`, `source`, `split`, `evaluation_split`, `method`, `n_samples`, and mean/std/median Dice and IoU. Per-sample details retain image and mask names plus foreground-pixel counts.

## Frontend model

Streamlit is used because it keeps the academic demonstration in one Python repository and can render images, tables, charts, and downloads without a separate API service. The app uses cached benchmark reads and cached checkpoint loading. It provides live inference only.

Training is a separate batch workflow:

```text
scripts/train_unet.py → model/unet_skin_lesion.keras
scripts/run_benchmark.py → results/*.csv, results/*.json, results/figures/*
app.py → reads those artifacts and analyzes uploads
```

## Operational concerns

- Dataset archives and model files are ignored by Git.
- Archive extraction rejects path traversal entries.
- Partial downloads are removed on transfer failure.
- Duplicate normalized image/mask stems fail loudly.
- Missing benchmark or checkpoint files produce visible UI messages.
- Uploads are processed in memory and are not persisted by the application.
- The project has no authentication, patient database, or clinical integration.
- Reports intentionally identify image descriptors as non-diagnostic.

## Deployment model

The canonical development environment is a Python virtual environment. Streamlit Community Cloud or another host can be considered only after a versioned model-artifact strategy, dependency validation, cold-start testing, and privacy review. Public deployment is not part of the current implementation.

## Risks and open questions

- CPU U-Net training on all 900 images can be slow; GPU execution is recommended for the main run.
- A small official segmentation dataset is useful for reproducibility but does not establish clinical generalisation.
- The best classical method may outperform an under-trained U-Net; the app therefore selects from measured metrics instead of assuming U-Net superiority.
- Future work could add augmentations, cross-validation, calibration, and stronger boundary metrics, but those are outside the current v1 contract.
