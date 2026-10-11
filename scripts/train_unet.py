"""Train and save the compact U-Net on a verified ISIC pair split."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from analyzer.data_loader import deterministic_split, download_dataset, pair_image_masks
from analyzer.metrics import dice_score, iou_score
from analyzer.preprocessing import resize_image
from analyzer.reporting import PIPELINE_VERSION
from analyzer.unet_segmentation import build_unet, compile_unet, load_unet_model


def _load_pairs(pairs, size=(256, 256)):
    images, masks = [], []
    for image_path, mask_path in pairs:
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        images.append(resize_image(image, size).astype("float32") / 255.0)
        resized_mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
        masks.append((resized_mask > 0).astype("float32")[..., None])
    return np.stack(images), np.stack(masks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["official_isic", "kaggle", "kagglehub"], default="official_isic")
    parser.add_argument("--cache-dir", default="dataset_cache")
    parser.add_argument("--num-samples", default="all", help="20, 100, 500, or all")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="model/unet_skin_lesion.keras")
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")

    image_dir, mask_dir, metadata = download_dataset(args.source, args.cache_dir)
    pairs = pair_image_masks(image_dir, mask_dir)
    if args.num_samples.lower() != "all":
        limit = int(args.num_samples)
        if limit < 3:
            parser.error("--num-samples must be at least 3")
        pairs = pairs[:limit]
    train_pairs, val_pairs, test_pairs = deterministic_split(pairs, seed=args.seed)
    train_images, train_masks = _load_pairs(train_pairs)
    val_images, val_masks = _load_pairs(val_pairs)
    test_images, test_masks = _load_pairs(test_pairs)

    # Keras writes temporary files while creating .keras checkpoints.  Keep
    # those files inside the project so Windows sandbox temp policies do not
    # make an otherwise valid training run fail.
    temp_dir = Path(".training_tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(temp_dir.resolve())
    os.environ["TEMP"] = str(temp_dir.resolve())
    os.environ["TMPDIR"] = str(temp_dir.resolve())

    import tensorflow as tf

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    model = compile_unet(build_unet((256, 256, 3)))
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    ]
    history = model.fit(
        train_images,
        train_masks,
        validation_data=(val_images, val_masks),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    best_model = load_unet_model(model_path)
    predictions = best_model.predict(test_images, batch_size=args.batch_size, verbose=0)[..., 0] >= 0.5
    test_dice = [dice_score(pred, truth[..., 0]) for pred, truth in zip(predictions, test_masks)]
    test_iou = [iou_score(pred, truth[..., 0]) for pred, truth in zip(predictions, test_masks)]
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history.history).to_csv(results_dir / "training_history.csv", index=False)
    metadata_path = results_dir / "training_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": PIPELINE_VERSION,
                "dataset": metadata,
                "seed": args.seed,
                "requested_samples": args.num_samples,
                "train_pairs": len(train_pairs),
                "validation_pairs": len(val_pairs),
                "test_pairs": len(test_pairs),
                "epochs_requested": args.epochs,
                "epochs_completed": len(history.history.get("loss", [])),
                "batch_size": args.batch_size,
                "model_path": str(model_path),
                "test_mean_dice": float(np.mean(test_dice)),
                "test_mean_iou": float(np.mean(test_iou)),
                "best_validation_loss": float(min(history.history["val_loss"])),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(
        {
            "dataset": metadata.get("dataset", "ISIC2018_Task1"),
            "source": metadata.get("source", args.source),
            "train_pairs": len(train_pairs),
            "validation_pairs": len(val_pairs),
            "test_pairs": len(test_pairs),
            "best_validation_loss": float(min(history.history["val_loss"])),
            "test_mean_dice": float(np.mean(test_dice)),
            "test_mean_iou": float(np.mean(test_iou)),
            "model_path": str(model_path),
        }
    )


if __name__ == "__main__":
    main()
