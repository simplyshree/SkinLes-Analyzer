"""Classical segmentation baselines with polarity and component safeguards."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import convert_color_space, ensure_rgb, gaussian_blur, to_grayscale


def _largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _candidate_score(mask: np.ndarray) -> float:
    """Score plausible lesion geometry without assuming a dark/light polarity."""
    binary = (mask > 0).astype(np.uint8)
    area = float(binary.mean())
    if area <= 0.002 or area >= 0.9:
        return -10.0
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    border_fraction = float(border.mean())
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return -10.0
    extent = ((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)) / binary.size
    return (1.0 - min(border_fraction, 1.0)) + 0.5 * min(extent, 1.0) - 0.25 * abs(area - 0.25)


def _cleanup(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = _largest_component(cleaned)
    flood = (cleaned * 255).astype(np.uint8)
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    flood_filled = flood.copy()
    cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood_filled)
    return ((flood | holes) > 0).astype(np.uint8)


def _choose_polarity(binary: np.ndarray) -> np.ndarray:
    candidates = [_largest_component(binary), _largest_component(1 - binary)]
    return max(candidates, key=_candidate_score).astype(np.uint8)


def otsu_segmentation(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    rgb = ensure_rgb(image)
    gray = gaussian_blur(rgb, 5)
    gray = to_grayscale(gray)
    _, thresholded = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    selected = _choose_polarity(thresholded)
    return _cleanup(selected, kernel_size)


def kmeans_segmentation(
    image: np.ndarray,
    k: int = 2,
    color_space: str = "LAB",
    random_state: int = 42,
    kernel_size: int = 5,
) -> np.ndarray:
    if k < 2:
        raise ValueError("k must be at least 2")
    representation = convert_color_space(ensure_rgb(image), color_space)
    pixels = representation.reshape((-1, representation.shape[-1])).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2)
    cv2.setRNGSeed(random_state)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    label_image = labels.reshape(representation.shape[:2])
    candidates = []
    for cluster_id in range(k):
        candidates.append(_largest_component((label_image == cluster_id).astype(np.uint8)))
    selected = max(candidates, key=_candidate_score)
    return _cleanup(selected, kernel_size)

