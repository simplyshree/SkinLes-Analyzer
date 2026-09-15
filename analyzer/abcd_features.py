"""Descriptive, non-diagnostic ABCD-style image features."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

from .preprocessing import ensure_rgb


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("The lesion mask must be a 2D array")
    return (array > 0).astype(np.uint8)


def asymmetry_score(mask: np.ndarray) -> float:
    binary = _clean_mask(mask)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0.0
    crop = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    side = max(crop.shape)
    square = np.zeros((side, side), dtype=np.uint8)
    y0 = (side - crop.shape[0]) // 2
    x0 = (side - crop.shape[1]) // 2
    square[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
    horizontal = np.flip(square, axis=1)
    vertical = np.flip(square, axis=0)
    denom = max(float(square.sum()), 1.0)
    difference = max(np.logical_xor(square, horizontal).sum(), np.logical_xor(square, vertical).sum())
    return float(np.clip(difference / denom, 0.0, 1.0))


def _contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours((_clean_mask(mask) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def border_features(mask: np.ndarray) -> dict[str, float]:
    binary = _clean_mask(mask)
    area = float(binary.sum())
    contour = _contour(binary)
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    circularity = float(4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
    circularity = float(np.clip(circularity, 0.0, 1.0))
    return {
        "area_pixels": area,
        "perimeter_pixels": perimeter,
        "circularity": circularity,
        "border_irregularity": float(1.0 - circularity),
    }


def colour_features(image: np.ndarray, mask: np.ndarray, n_clusters: int = 3) -> dict[str, object]:
    rgb = ensure_rgb(image)
    binary = _clean_mask(mask)
    if rgb.shape[:2] != binary.shape:
        raise ValueError("Image and mask dimensions differ")
    pixels = rgb[binary > 0].astype(np.float32)
    if len(pixels) == 0:
        return {
            "mean_rgb": [0.0, 0.0, 0.0],
            "std_rgb": [0.0, 0.0, 0.0],
            "colour_variation": 0.0,
            "dominant_colours": [],
        }
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    variation = float(np.mean(std / 255.0))
    count = min(max(1, n_clusters), len(pixels))
    model = KMeans(n_clusters=count, random_state=42, n_init=5)
    model.fit(pixels)
    labels, counts = np.unique(model.labels_, return_counts=True)
    order = labels[np.argsort(-counts)]
    dominant = [model.cluster_centers_[int(i)].round(2).tolist() for i in order]
    return {
        "mean_rgb": mean.round(3).tolist(),
        "std_rgb": std.round(3).tolist(),
        "colour_variation": variation,
        "dominant_colours": dominant,
    }


def diameter_pixels(mask: np.ndarray) -> float:
    binary = _clean_mask(mask)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0.0
    return float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))


def extract_abcd_features(image: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    border = border_features(mask)
    colour = colour_features(image, mask)
    return {
        "asymmetry": asymmetry_score(mask),
        "border_irregularity": border["border_irregularity"],
        "circularity": border["circularity"],
        "area_pixels": border["area_pixels"],
        "perimeter_pixels": border["perimeter_pixels"],
        "colour_variation": colour["colour_variation"],
        "mean_rgb": colour["mean_rgb"],
        "std_rgb": colour["std_rgb"],
        "dominant_colours": colour["dominant_colours"],
        "diameter_pixels": diameter_pixels(mask),
    }

