"""Remote ISIC 2018 archive loading with explicit pairing checks."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ISIC2018_URLS = {
    "training_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "training_masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "validation_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "validation_masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
}


def download_and_extract(url: str, cache_dir: str | Path, archive_name: str) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / archive_name
    extract_dir = cache / archive.stem
    if not extract_dir.exists():
        if not archive.exists():
            urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)
    return extract_dir


def download_isic2018(cache_dir: str | Path, split: str = "training") -> tuple[Path, Path]:
    """Download one split, returning image and mask directories."""
    split = split.lower()
    if split not in {"training", "validation"}:
        raise ValueError("split must be training or validation")
    image_key = f"{split}_images"
    mask_key = f"{split}_masks"
    image_dir = download_and_extract(ISIC2018_URLS[image_key], cache_dir, f"{image_key}.zip")
    mask_dir = download_and_extract(ISIC2018_URLS[mask_key], cache_dir, f"{mask_key}.zip")
    return image_dir, mask_dir


def _index_files(directory: str | Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    root = Path(directory)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    return {p.stem: p for p in files}


def pair_image_masks(image_dir: str | Path, mask_dir: str | Path) -> list[tuple[Path, Path]]:
    images = _index_files(image_dir, (".jpg", ".jpeg", ".png"))
    masks = _index_files(mask_dir, (".png", ".jpg", ".jpeg"))
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks or missing_images:
        raise ValueError(f"Image/mask mismatch: {len(missing_masks)} images lack masks; {len(missing_images)} masks lack images")
    pairs = sorted((images[key], masks[key]) for key in images)
    if not pairs:
        raise ValueError("No image/mask pairs found")
    return pairs


def load_pair(image_path: str | Path, mask_path: str | Path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    return image, mask
