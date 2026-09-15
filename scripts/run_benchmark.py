"""Evaluate all available segmentation methods on one matched evaluation set."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd
from PIL import Image

from analyzer.classical_segmentation import kmeans_segmentation, otsu_segmentation
from analyzer.data_loader import pair_image_masks
from analyzer.metrics import dice_score, iou_score, summarize_scores
from analyzer.preprocessing import resize_image


def _read_pair(image_path: Path, mask_path: Path, size=(256, 256)):
    image = np.asarray(Image.open(image_path).convert("RGB"))
    mask = np.asarray(Image.open(mask_path).convert("L"))
    image = resize_image(image, size)
    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    return image, (mask > 0).astype(np.uint8)


def evaluate_method(pairs, method: str, model=None):
    ious, dices = [], []
    for image_path, mask_path in pairs:
        image, truth = _read_pair(image_path, mask_path)
        if method == "Otsu":
            prediction = otsu_segmentation(image)
        elif method == "K-means + Morphology":
            prediction = kmeans_segmentation(image)
        elif method == "U-Net":
            from analyzer.unet_segmentation import predict_unet

            prediction = predict_unet(model, image)
        else:
            raise ValueError(method)
        ious.append(iou_score(prediction, truth))
        dices.append(dice_score(prediction, truth))
    return summarize_scores(ious, dices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--output", default="results/benchmark_results.csv")
    args = parser.parse_args()
    pairs = pair_image_masks(args.images_dir, args.masks_dir)[: args.num_samples]
    methods = [("Otsu", None), ("K-means + Morphology", None)]
    if args.model_path:
        from analyzer.unet_segmentation import load_unet_model

        methods.append(("U-Net", load_unet_model(args.model_path)))
    rows = []
    for method, model in methods:
        summary = evaluate_method(pairs, method, model)
        rows.append({"dataset": "ISIC2018_Task1", "source": "executed benchmark", "method": method, **summary})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
