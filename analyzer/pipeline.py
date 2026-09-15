"""One reusable analysis entry point for the app and downstream notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .abcd_features import extract_abcd_features
from .classical_segmentation import kmeans_segmentation, otsu_segmentation
from .preprocessing import ensure_rgb, preprocess_image
from .visualization import contour_overlay, overlay_mask


@dataclass(frozen=True)
class AnalysisConfig:
    target_size: tuple[int, int] = (256, 256)
    remove_hair: bool = False
    use_clahe: bool = False
    benchmark_path: str = "results/benchmark_results.csv"


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def select_best_method(benchmark_path: str | Path = "results/benchmark_results.csv") -> str:
    try:
        import pandas as pd

        table = pd.read_csv(_resolve_path(benchmark_path))
        required = {"method", "mean_dice", "mean_iou"}
        if not required.issubset(table.columns):
            return "otsu"
        valid = table.dropna(subset=["mean_dice", "mean_iou"])
        if "dataset" in valid.columns:
            # A local synthetic regression fixture verifies code paths but is
            # not evidence for choosing a production segmentation method.
            isic_rows = valid[valid["dataset"].astype(str).str.contains("ISIC", case=False, na=False)]
            if isic_rows.empty:
                return "otsu"
            valid = isic_rows
        if valid.empty:
            return "otsu"
        row = valid.sort_values(["mean_dice", "mean_iou"], ascending=False).iloc[0]
        return str(row["method"]).strip().lower()
    except (FileNotFoundError, OSError, ValueError, ImportError):
        return "otsu"


def analyze_lesion(
    image: np.ndarray,
    method: str | None = None,
    model=None,
    config: AnalysisConfig | None = None,
) -> dict[str, object]:
    """Analyze one RGB image, returning images, mask, overlay, and ABCD features."""
    cfg = config or AnalysisConfig()
    original = ensure_rgb(image)
    prepared = preprocess_image(original, cfg.target_size, cfg.remove_hair, cfg.use_clahe)
    selected = (method or select_best_method(cfg.benchmark_path)).lower()
    warning = None
    if selected == "otsu":
        mask = otsu_segmentation(prepared["processed"])
    elif selected in {"kmeans", "k-means", "k-means + morphology", "kmeans + morphology"}:
        mask = kmeans_segmentation(prepared["processed"])
        selected = "kmeans"
    elif selected in {"u-net", "unet"}:
        if model is None:
            warning = "The benchmark selected U-Net, but no trained model was provided; Otsu is used for this request."
            mask = otsu_segmentation(prepared["processed"])
            selected = "otsu_fallback"
        else:
            from .unet_segmentation import predict_unet

            mask = predict_unet(model, prepared["processed"])
            selected = "unet"
    else:
        raise ValueError(f"Unknown segmentation method: {method}")
    abcd = extract_abcd_features(prepared["processed"], mask)
    result = {
        "original": original,
        "preprocessed": prepared["processed"],
        "mask": mask.astype(np.uint8),
        "contour_overlay": contour_overlay(prepared["processed"], mask),
        "mask_overlay": overlay_mask(prepared["processed"], mask),
        "selected_method": selected,
        "warning": warning,
        "hair_mask": prepared["hair_mask"],
        "processing_steps": [
            "Original image",
            "Resize and RGB normalization",
            "Optional hair/artifact removal and local contrast adjustment",
            f"Segmentation: {selected}",
            "Morphological cleanup",
            "Contour extraction",
            "ABCD-style image feature extraction",
        ],
    }
    result.update(abcd)
    result["abcd"] = abcd
    return result
