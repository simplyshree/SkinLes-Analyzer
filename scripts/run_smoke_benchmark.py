"""Run a deterministic synthetic regression benchmark; this is not ISIC performance."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd

from analyzer.classical_segmentation import kmeans_segmentation, otsu_segmentation
from analyzer.metrics import dice_score, iou_score, summarize_scores


def fixture(index: int, size: int = 128):
    rng = np.random.default_rng(index)
    image = np.full((size, size, 3), [205, 175, 155], dtype=np.uint8)
    truth = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2 + int(rng.integers(-8, 9)), size // 2 + int(rng.integers(-8, 9)))
    axes = (int(rng.integers(18, 30)), int(rng.integers(14, 26)))
    cv2.ellipse(truth, center, axes, int(rng.integers(0, 180)), 0, 360, 1, -1)
    image[truth > 0] = [75, 55, 48]
    noise = rng.normal(0, 5, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8), truth


def main():
    rows = []
    for method, fn in [("Otsu", otsu_segmentation), ("K-means + Morphology", kmeans_segmentation)]:
        ious, dices = [], []
        for index in range(8):
            image, truth = fixture(index)
            prediction = fn(image)
            ious.append(iou_score(prediction, truth))
            dices.append(dice_score(prediction, truth))
        rows.append({"dataset": "synthetic_smoke_fixture", "source": "executed regression smoke test", "method": method, **summarize_scores(ious, dices)})
    output = Path("results/benchmark_results.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
