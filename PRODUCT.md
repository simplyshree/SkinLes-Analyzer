# Product Brief

## Problem

Students and reviewers need a transparent way to understand how image-processing methods isolate a skin lesion, how a learned segmentation model differs from classical baselines, and how mask quality can be measured without presenting the output as a diagnosis.

## Target users

- Students presenting a computer-vision or medical-image-processing project.
- Instructors/reviewers evaluating reproducibility and algorithmic reasoning.
- Developers experimenting with segmentation baselines and a compact U-Net.

## Goals

- Make the complete image-processing pipeline understandable.
- Compare Otsu, K-means plus morphology, and U-Net on identical held-out samples.
- Use all 900 practical official image/mask pairs for the main experiment.
- Provide a polished but simple Streamlit demonstration.
- Make every result reproducible from a command and saved metadata.
- Keep the medical scope explicitly educational and non-diagnostic.

## Non-goals

- Disease classification or melanoma diagnosis.
- Clinical risk scoring or treatment recommendations.
- Automatic training from the frontend.
- Public deployment before local validation.
- Claiming that ABCD-style image descriptors are clinical ABCD criteria.

## Main workflows

### Research workflow

1. Create/activate the virtual environment.
2. Download and verify the official dataset.
3. Train the U-Net offline.
4. Run the all-900 benchmark.
5. Inspect CSV, split manifest, JSON metadata, per-sample metrics, and figures.

### Presentation workflow

1. Start Streamlit.
2. Open the “How it works” tab to introduce the theory.
3. Upload a representative dermoscopic image.
4. Compare method outputs and the contour overlay.
5. Explain the measured benchmark ranking.
6. Show the ABCD-style descriptors with the non-diagnostic disclaimer.

### Debug workflow

Use 20 samples, run tests, inspect the saved manifest and figures, then scale to 100, 500, and all samples.

## Success criteria

- A new user can install, run, test, benchmark, and launch the app from the README.
- The app responds to a valid upload without training.
- Invalid images, missing checkpoints, and missing benchmark files produce actionable messages.
- All methods are compared on the same deterministic test set.
- Real measured metrics replace illustrative values.
- The main experiment produces the documented artifact set.
- A reviewer can explain every processing stage from the README and frontend.

## Constraints

- Python and Streamlit remain the implementation stack.
- The default practical dataset is the smaller official Task 1 bundle; large Kaggle/KaggleHub data stays optional.
- Dataset and trained model artifacts must remain outside Git.
- Experiments must be deterministic through explicit seeds and manifests.
- Output language must avoid diagnostic claims.
