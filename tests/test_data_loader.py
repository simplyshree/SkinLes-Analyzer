from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from analyzer.data_loader import deterministic_split, download_and_extract, pair_image_masks


def _write_image(path: Path, value: int):
    Image.fromarray(np.full((8, 8, 3), value, dtype=np.uint8)).save(path)


def _write_mask(path: Path):
    Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255).save(path)


def test_pairing_normalizes_segmentation_suffix(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_image(images / "ISIC_0000001.jpg", 50)
    _write_mask(masks / "ISIC_0000001_segmentation.png")
    pairs = pair_image_masks(images, masks)
    assert len(pairs) == 1
    assert pairs[0][0].stem == "ISIC_0000001"


def test_pairing_rejects_unmatched_files(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    _write_image(images / "ISIC_0000001.jpg", 50)
    _write_mask(masks / "ISIC_0000002_segmentation.png")
    with pytest.raises(ValueError, match="Image/mask mismatch"):
        pair_image_masks(images, masks)


def test_deterministic_split_is_repeatable(tmp_path):
    pairs = [(tmp_path / f"image_{i}.jpg", tmp_path / f"mask_{i}.png") for i in range(10)]
    first = deterministic_split(pairs, seed=7)
    second = deterministic_split(pairs, seed=7)
    assert first == second
    assert sum(len(part) for part in first) == len(pairs)
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[1]).isdisjoint(first[2])


def test_corrupt_cached_archive_is_rejected(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "broken.zip").write_bytes(b"not a zip archive")
    with pytest.raises(ValueError, match="not a valid ZIP"):
        download_and_extract("https://example.invalid/broken.zip", cache, "broken.zip")

