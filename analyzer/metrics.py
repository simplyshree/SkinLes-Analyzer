"""Segmentation metrics independent from any model implementation."""

from __future__ import annotations

import numpy as np


def _binary_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction) > 0
    truth = np.asarray(target) > 0
    if pred.shape != truth.shape:
        raise ValueError(f"Mask shapes differ: {pred.shape} vs {truth.shape}")
    return pred, truth


def iou_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _binary_pair(prediction, target)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _binary_pair(prediction, target)
    intersection = np.logical_and(pred, truth).sum()
    denominator = pred.sum() + truth.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def summarize_scores(iou_values: list[float], dice_values: list[float]) -> dict[str, float | int]:
    if not iou_values or not dice_values:
        raise ValueError("Cannot summarize an empty evaluation")
    return {
        "n_samples": len(iou_values),
        "mean_iou": float(np.mean(iou_values)),
        "std_iou": float(np.std(iou_values)),
        "median_iou": float(np.median(iou_values)),
        "mean_dice": float(np.mean(dice_values)),
        "std_dice": float(np.std(dice_values)),
        "median_dice": float(np.median(dice_values)),
    }

