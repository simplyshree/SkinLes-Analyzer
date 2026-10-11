"""Small, testable preprocessing operations used by the benchmark and app."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def ensure_rgb(image: np.ndarray, input_color_space: str = "RGB") -> np.ndarray:
    """Return a uint8 RGB image, accepting grayscale, RGB, or BGR arrays."""
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("Image cannot be empty")
    if array.ndim == 2:
        array = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("Expected an image shaped HxW, HxWx3, or HxWx4")
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if input_color_space.upper() == "BGR":
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif input_color_space.upper() != "RGB":
        raise ValueError("input_color_space must be RGB or BGR")
    return np.ascontiguousarray(array)


def resize_image(image: np.ndarray, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize to (height, width) while keeping an explicit interpolation rule."""
    rgb = ensure_rgb(image)
    height, width = size
    if len(size) != 2 or height <= 0 or width <= 0:
        raise ValueError("size must contain two positive integers: (height, width)")
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(ensure_rgb(image), cv2.COLOR_RGB2GRAY)


def gaussian_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return cv2.GaussianBlur(ensure_rgb(image), (kernel_size, kernel_size), 0)


def median_filter(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")
    return cv2.medianBlur(ensure_rgb(image), kernel_size)


def convert_color_space(image: np.ndarray, color_space: str) -> np.ndarray:
    rgb = ensure_rgb(image)
    conversions = {
        "RGB": lambda x: x,
        "GRAY": lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY),
        "HSV": lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2HSV),
        "LAB": lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2LAB),
    }
    key = color_space.upper()
    if key not in conversions:
        raise ValueError(f"Unsupported color space: {color_space}")
    return conversions[key](rgb)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """Enhance local contrast in the luminance channel without changing hue."""
    if clip_limit <= 0 or len(tile_grid_size) != 2 or min(tile_grid_size) <= 0:
        raise ValueError("clip_limit and tile_grid_size must be positive")
    rgb = ensure_rgb(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def remove_hair_artifacts(image: np.ndarray, kernel_size: int = 17, inpaint_radius: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Use black-hat morphology to detect dark hair-like structures, then inpaint."""
    if kernel_size <= 0 or kernel_size % 2 == 0 or inpaint_radius <= 0:
        raise ValueError("kernel_size must be positive odd and inpaint_radius must be positive")
    rgb = ensure_rgb(image)
    gray = to_grayscale(rgb)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(black_hat, 10, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    repaired = cv2.inpaint(rgb, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return repaired, hair_mask


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (256, 256),
    remove_hair: bool = False,
    use_clahe: bool = False,
) -> dict[str, Any]:
    """Create one consistent representation for classical methods and the app."""
    original = ensure_rgb(image)
    resized = resize_image(original, target_size)
    hair_mask = np.zeros(resized.shape[:2], dtype=np.uint8)
    processed = resized.copy()
    if remove_hair:
        processed, hair_mask = remove_hair_artifacts(processed)
    if use_clahe:
        processed = apply_clahe(processed)
    return {
        "original": original,
        "resized": resized,
        "processed": processed,
        "grayscale": to_grayscale(processed),
        "hsv": convert_color_space(processed, "HSV"),
        "lab": convert_color_space(processed, "LAB"),
        "hair_mask": hair_mask,
    }

