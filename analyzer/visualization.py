"""Presentation helpers shared by the notebook and Streamlit app."""

from __future__ import annotations

import cv2
import numpy as np

from .preprocessing import ensure_rgb


def mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    return ((np.asarray(mask) > 0).astype(np.uint8) * 255)


def overlay_mask(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (255, 80, 40), alpha: float = 0.35) -> np.ndarray:
    rgb = ensure_rgb(image).copy()
    colour_layer = np.zeros_like(rgb)
    colour_layer[:, :] = np.array(colour, dtype=np.uint8)
    foreground = np.asarray(mask) > 0
    if foreground.shape != rgb.shape[:2]:
        raise ValueError("Image and mask dimensions differ")
    rgb[foreground] = ((1 - alpha) * rgb[foreground] + alpha * colour_layer[foreground]).astype(np.uint8)
    return rgb


def contour_overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int] = (40, 220, 80)) -> np.ndarray:
    rgb = ensure_rgb(image).copy()
    contours, _ = cv2.findContours(mask_to_uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contours, -1, colour, 2)
    return rgb

