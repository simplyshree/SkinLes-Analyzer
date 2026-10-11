"""Remote ISIC 2018 archive loading with explicit pairing checks."""

from __future__ import annotations

import urllib.request
import zipfile
import json
import hashlib
import random
from pathlib import Path

from PIL import Image

ISIC2018_URLS = {
    "training_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "training_masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip",
    "validation_images": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "validation_masks": "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Validation_GroundTruth.zip",
}

# The ISIC archive's official small Task 1 bundle is the 2016 release:
# approximately 602 MB of JPEG images and 6 MB of binary masks.  The similarly
# named 2018 Task 1-2 training archive is over 10 GB, so it is intentionally
# not the default local workflow.
ISIC2016_URLS = {
    "training_images": "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_Data.zip",
    "training_masks": "https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip",
}

DEFAULT_KAGGLE_DATASET = "tschandl/isic2018-challenge-task1-data-segmentation"
DEFAULT_DATA_SOURCE = "official_isic"


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_metadata(cache_dir: Path, **metadata) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output = cache_dir / "dataset_metadata.json"
    existing = {}
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(metadata)
    output.write_text(json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8")
    return existing


def download_and_extract(url: str, cache_dir: str | Path, archive_name: str) -> Path:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / archive_name
    extract_dir = cache / archive.stem
    # Keep the archive and extraction independently recoverable.  A user may
    # delete either one to reclaim space; the next run should repair the cache
    # instead of failing while trying to hash a missing archive.
    if not archive.exists():
        partial = archive.with_suffix(archive.suffix + ".part")
        try:
            import requests

            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
        except ImportError:
            urllib.request.urlretrieve(url, partial)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(archive)
    if not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        raise ValueError(f"Downloaded archive is not a valid ZIP: {archive}")
    if not extract_dir.exists():
        with zipfile.ZipFile(archive) as zf:
            root = extract_dir.resolve()
            for member in zf.infolist():
                destination = (extract_dir / member.filename).resolve()
                if not destination.is_relative_to(root):
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            zf.extractall(extract_dir)
    metadata_path = cache / "dataset_metadata.json"
    existing = {}
    if metadata_path.exists():
        try:
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    archives = existing.get("archives", {})
    archives[archive_name] = {
        "url": url,
        "sha256": _sha256(archive),
        "bytes": archive.stat().st_size,
        "extract_dir": str(extract_dir),
    }
    _write_metadata(cache, cache_version=1, archives=archives)
    return extract_dir


def download_isic2018(cache_dir: str | Path, split: str = "training") -> tuple[Path, Path]:
    """Download one official ISIC split, returning image and mask directories."""
    split = split.lower()
    if split not in {"training", "validation"}:
        raise ValueError("split must be training or validation")
    image_key = f"{split}_images"
    mask_key = f"{split}_masks"
    image_dir = download_and_extract(ISIC2018_URLS[image_key], cache_dir, f"{image_key}.zip")
    mask_dir = download_and_extract(ISIC2018_URLS[mask_key], cache_dir, f"{mask_key}.zip")
    pairs = pair_image_masks(image_dir, mask_dir)
    _write_metadata(Path(cache_dir), source=DEFAULT_DATA_SOURCE, split=split, n_pairs=len(pairs))
    return image_dir, mask_dir


def download_isic2016(cache_dir: str | Path) -> tuple[Path, Path]:
    """Download the official small ISIC Task 1 training split."""
    image_dir = download_and_extract(
        ISIC2016_URLS["training_images"], cache_dir, "isic2016_training_images.zip"
    )
    mask_dir = download_and_extract(
        ISIC2016_URLS["training_masks"], cache_dir, "isic2016_training_masks.zip"
    )
    pairs = pair_image_masks(image_dir, mask_dir)
    _write_metadata(
        Path(cache_dir),
        source=DEFAULT_DATA_SOURCE,
        dataset="ISIC2016_Task1",
        split="training",
        n_pairs=len(pairs),
    )
    return image_dir, mask_dir


def download_dataset(
    source: str = DEFAULT_DATA_SOURCE,
    cache_dir: str | Path = "dataset_cache",
    split: str = "training",
) -> tuple[Path, Path, dict]:
    """Download a configured dataset source and return folders plus metadata."""
    source = source.lower().strip()
    cache = Path(cache_dir)
    if source == "official_isic":
        if split.lower() != "training":
            raise ValueError("the small official_isic source provides only the training split")
        image_dir, mask_dir = download_isic2016(cache)
    elif source == "kaggle":
        image_dir, mask_dir = download_isic2018_kaggle(cache, split=split)
    elif source == "kagglehub":
        _, image_dir, mask_dir = download_isic2018_kagglehub(split=split)
    else:
        raise ValueError("source must be official_isic, kaggle, or kagglehub")
    pairs = pair_image_masks(image_dir, mask_dir)
    metadata = _write_metadata(
        cache,
        source=source,
        dataset="ISIC2016_Task1" if source == DEFAULT_DATA_SOURCE else "ISIC2018_Task1",
        split=split,
        n_pairs=len(pairs),
    )
    return image_dir, mask_dir, metadata


def download_isic2018_kaggle(
    cache_dir: str | Path,
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    split: str = "training",
) -> tuple[Path, Path]:
    """Download the Kaggle ISIC 2018 segmentation dataset and locate one split.

    Kaggle authentication is handled by the official `kaggle` package using
    `~/.kaggle/kaggle.json` or KAGGLE_USERNAME/KAGGLE_KEY environment variables.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise ImportError("Install the Kaggle API with: pip install kaggle datasets") from exc
    split = split.lower()
    if split not in {"training", "validation"}:
        raise ValueError("split must be training or validation")
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    dataset_dir = cache / "kaggle_isic2018_task1"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(dataset_dir), unzip=True, quiet=False)
    return _locate_split_dirs(dataset_dir, split)


def locate_isic2018_split(dataset_root: str | Path, split: str = "training") -> tuple[Path, Path]:
    """Locate matched image and mask directories in a kagglehub result path."""
    split = split.lower()
    if split not in {"training", "validation"}:
        raise ValueError("split must be training or validation")
    return _locate_split_dirs(Path(dataset_root), split)


def download_isic2018_kagglehub(
    dataset_slug: str = DEFAULT_KAGGLE_DATASET,
    split: str = "training",
) -> tuple[str, Path, Path]:
    """Download with kagglehub and return the cache path plus matched folders."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Install kagglehub with: pip install kagglehub") from exc
    path = kagglehub.dataset_download(dataset_slug)
    image_dir, mask_dir = locate_isic2018_split(path, split)
    return path, image_dir, mask_dir


def _index_files(directory: str | Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    root = Path(directory)
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes]
    indexed = {}
    for path in files:
        stem = path.stem.lower()
        for suffix in ("_segmentation", "_mask", "_masks"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
        if stem in indexed and indexed[stem] != path:
            raise ValueError(f"Duplicate normalized filename stem '{stem}' in {root}")
        indexed[stem] = path
    return indexed


def _locate_split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    """Find image and ground-truth directories across common Kaggle layouts."""
    split_token = "train" if split == "training" else "val"
    directories = [root, *[p for p in root.rglob("*") if p.is_dir()]]
    image_candidates = []
    mask_candidates = []
    for directory in directories:
        name = directory.name.lower()
        files = list(directory.glob("*"))
        if any(p.suffix.lower() in {".jpg", ".jpeg"} for p in files):
            score = int("input" in name or "image" in name) + int(split_token in name)
            image_candidates.append((score, directory))
        if any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in files):
            score = int(any(token in name for token in ("mask", "ground", "segment"))) + int(split_token in name)
            mask_candidates.append((score, directory))
    image_candidates.sort(key=lambda item: item[0], reverse=True)
    mask_candidates.sort(key=lambda item: item[0], reverse=True)
    for _, image_dir in image_candidates:
        for _, mask_dir in mask_candidates:
            if image_dir == mask_dir:
                continue
            try:
                pair_image_masks(image_dir, mask_dir)
                return image_dir, mask_dir
            except ValueError:
                continue
    raise FileNotFoundError(f"Could not locate matched {split} image and mask directories under {root}")


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


def deterministic_split(
    pairs: list[tuple[Path, Path]], seed: int = 42
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Return deterministic train, validation, and test pair lists."""
    if len(pairs) < 3:
        raise ValueError("At least 3 image/mask pairs are required for a split")
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * 0.2))
    n_val = max(1, round(len(shuffled) * 0.2))
    if n_test + n_val >= len(shuffled):
        n_test = 1
        n_val = 1
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    return train, val, test


def load_pair(image_path: str | Path, mask_path: str | Path):
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    return image, mask
