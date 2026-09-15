import cv2
import numpy as np

from analyzer.classical_segmentation import kmeans_segmentation, otsu_segmentation
from analyzer.metrics import dice_score, iou_score


def synthetic_lesion():
    image = np.full((96, 96, 3), [210, 180, 160], dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    cv2.ellipse(mask, (48, 48), (25, 18), 15, 0, 360, 1, -1)
    image[mask > 0] = [65, 45, 40]
    return image, mask


def test_classical_methods_return_binary_masks():
    image, _ = synthetic_lesion()
    for method in (otsu_segmentation, kmeans_segmentation):
        mask = method(image)
        assert mask.shape == image.shape[:2]
        assert set(np.unique(mask)).issubset({0, 1})


def test_metrics_known_values_and_empty_edge_case():
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert iou_score(pred, truth) == 0.5
    assert dice_score(pred, truth) == 2 / 3
    empty = np.zeros((2, 2), dtype=np.uint8)
    assert iou_score(empty, empty) == 1.0
    assert dice_score(empty, empty) == 1.0

