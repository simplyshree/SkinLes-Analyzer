"""Run a reproducible segmentation benchmark and save presentation artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from analyzer.classical_segmentation import kmeans_segmentation, otsu_segmentation
from analyzer.data_loader import (
    DEFAULT_DATA_SOURCE,
    deterministic_split,
    download_dataset,
    pair_image_masks,
)
from analyzer.metrics import dice_score, iou_score, summarize_scores
from analyzer.preprocessing import resize_image
from analyzer.reporting import PIPELINE_VERSION


def _read_pair(image_path: Path, mask_path: Path, size=(256, 256)):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    image = resize_image(image, size)
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return image, (mask > 0).astype(np.uint8)


def _predict(method: str, image: np.ndarray, model=None) -> np.ndarray:
    if method == "Otsu":
        return otsu_segmentation(image)
    if method == "K-means + Morphology":
        return kmeans_segmentation(image)
    if method == "U-Net":
        from analyzer.unet_segmentation import predict_unet

        return predict_unet(model, image)
    raise ValueError(method)


def evaluate_method(pairs, method: str, model=None):
    """Return aggregate metrics, per-sample metrics, and predictions."""
    ious, dices, details, predictions = [], [], [], []
    if method == "U-Net":
        images, truths = zip(*[_read_pair(image, mask) for image, mask in pairs])
        batch = np.stack(images).astype(np.float32) / 255.0
        raw_predictions = model.predict(batch, batch_size=8, verbose=0)[..., 0]
        predictions = [(raw >= 0.5).astype(np.uint8) for raw in raw_predictions]
        loaded = list(zip(images, truths))
    else:
        loaded = [_read_pair(image, mask) for image, mask in pairs]
        predictions = [_predict(method, image) for image, _ in loaded]
    for (image_path, mask_path), (_, truth), prediction in zip(pairs, loaded, predictions):
        iou = iou_score(prediction, truth)
        dice = dice_score(prediction, truth)
        ious.append(iou)
        dices.append(dice)
        details.append(
            {
                "image": image_path.name,
                "mask": mask_path.name,
                "iou": iou,
                "dice": dice,
                "prediction_foreground_pixels": int(np.asarray(prediction).sum()),
                "truth_foreground_pixels": int(np.asarray(truth).sum()),
            }
        )
    return summarize_scores(ious, dices), details, predictions


def _save_split_manifest(pairs, train_pairs, val_pairs, test_pairs, path: Path):
    split_by_pair = {}
    for split, group in (("train", train_pairs), ("validation", val_pairs), ("test", test_pairs)):
        for pair in group:
            split_by_pair[(str(pair[0]), str(pair[1]))] = split
    rows = []
    for index, (image_path, mask_path) in enumerate(pairs):
        rows.append(
            {
                "index": index,
                "split": split_by_pair[(str(image_path), str(mask_path))],
                "image": str(image_path),
                "mask": str(mask_path),
                "image_id": image_path.stem.lower(),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_figures(test_pairs, prediction_map, output_dir: Path):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    method_names = list(prediction_map)
    sample_count = min(4, len(test_pairs))
    columns = 2 + len(method_names)
    figure, axes = plt.subplots(sample_count, columns, figsize=(4 * columns, 4 * sample_count), squeeze=False)
    for row, pair in enumerate(test_pairs[:sample_count]):
        image, truth = _read_pair(*pair)
        views = [(image, "Original", None), (truth, "Ground truth", "gray")]
        for method in method_names:
            prediction = prediction_map[method][row]
            views.append((prediction, method, "gray"))
        for column, (view, title, cmap) in enumerate(views):
            axes[row, column].imshow(view, cmap=cmap)
            axes[row, column].set_title(title)
            axes[row, column].axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / "comparison_grid.png", dpi=160)
    plt.close(figure)


def _save_metric_figure(summary_rows, output_dir: Path):
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary_rows)
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(frame))
    width = 0.35
    axis.bar(x - width / 2, frame["mean_dice"], width, label="Mean Dice")
    axis.bar(x + width / 2, frame["mean_iou"], width, label="Mean IoU")
    axis.set_xticks(x, frame["method"], rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("Held-out test performance")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "metric_summary.png", dpi=160)
    plt.close(figure)


def _parse_limit(value: str, total: int) -> int:
    if value.lower() == "all":
        return total
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError("--num-samples must be 20, 100, 500, or all") from exc
    if limit <= 0:
        raise ValueError("--num-samples must be positive or all")
    return min(limit, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["official_isic", "kaggle", "kagglehub"], default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--cache-dir", default="dataset_cache")
    parser.add_argument("--images-dir")
    parser.add_argument("--masks-dir")
    parser.add_argument("--num-samples", default="all", help="20, 100, 500, or all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="results/benchmark_results.csv")
    parser.add_argument("--manifest", default="results/split_manifest.csv")
    parser.add_argument("--metadata", default="results/run_metadata.json")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()
    if bool(args.images_dir) != bool(args.masks_dir):
        parser.error("--images-dir and --masks-dir must be supplied together")
    try:
        if args.images_dir:
            pairs = pair_image_masks(args.images_dir, args.masks_dir)
            dataset_metadata = {"source": "local_paths", "dataset": "local_paths", "split": "unknown", "n_pairs": len(pairs)}
        else:
            image_dir, mask_dir, dataset_metadata = download_dataset(args.source, args.cache_dir)
            pairs = pair_image_masks(image_dir, mask_dir)
        pairs = pairs[: _parse_limit(args.num_samples, len(pairs))]
        if len(pairs) < 3:
            parser.error("at least 3 matched pairs are required for the deterministic split")
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    train_pairs, val_pairs, test_pairs = deterministic_split(pairs, seed=args.seed)
    _save_split_manifest(pairs, train_pairs, val_pairs, test_pairs, Path(args.manifest))
    methods = [("Otsu", None), ("K-means + Morphology", None)]
    if args.model_path:
        from analyzer.unet_segmentation import load_unet_model

        model = load_unet_model(args.model_path)
        methods.append(("U-Net", model))
    summary_rows, detail_rows, prediction_map = [], [], {}
    for method, method_model in methods:
        summary, details, predictions = evaluate_method(test_pairs, method, method_model)
        prediction_map[method] = predictions
        detail_rows.extend({"method": method, **detail} for detail in details)
        summary_rows.append(
            {
                "dataset": dataset_metadata.get("dataset", "ISIC2018_Task1"),
                "source": dataset_metadata["source"],
                "split": dataset_metadata["split"],
                "evaluation_split": "test",
                "method": method,
                **summary,
            }
        )
    results = pd.DataFrame(summary_rows).sort_values(["mean_dice", "mean_iou"], ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    details_path = output.with_name(f"{output.stem}_per_sample.csv")
    pd.DataFrame(detail_rows).to_csv(details_path, index=False)
    figures_dir = Path(args.figures_dir)
    _save_figures(test_pairs, prediction_map, figures_dir)
    _save_metric_figure(summary_rows, figures_dir)
    metadata = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "source": dataset_metadata,
        "seed": args.seed,
        "requested_samples": args.num_samples,
        "selected_pairs": len(pairs),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "methods": [method for method, _ in methods],
        "model_path": args.model_path,
        "benchmark_csv": str(output),
        "per_sample_csv": str(details_path),
        "figures_dir": str(figures_dir),
        "selected_best_method": str(results.iloc[0]["method"]),
    }
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps({"manifest": args.manifest, "metadata": args.metadata, "figures": args.figures_dir}, indent=2))


if __name__ == "__main__":
    main()
