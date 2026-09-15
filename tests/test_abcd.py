import cv2
import numpy as np

from analyzer.abcd_features import asymmetry_score, border_features, colour_features, diameter_pixels, extract_abcd_features


def circle_mask(size=80):
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 4, 1, -1)
    return mask


def test_symmetric_mask_has_low_asymmetry_and_valid_border():
    mask = circle_mask()
    assert asymmetry_score(mask) < 0.05
    features = border_features(mask)
    assert features["area_pixels"] > 0
    assert 0 <= features["circularity"] <= 1
    assert 0 <= features["border_irregularity"] <= 1
    assert diameter_pixels(mask) > 0


def test_colour_features_are_masked_and_structured():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:] = [220, 220, 220]
    image[circle_mask() > 0] = [80, 40, 30]
    features = colour_features(image, circle_mask())
    assert len(features["mean_rgb"]) == 3
    assert len(features["dominant_colours"]) >= 1
    all_features = extract_abcd_features(image, circle_mask())
    assert set(["asymmetry", "border_irregularity", "colour_variation", "diameter_pixels"]).issubset(all_features)

