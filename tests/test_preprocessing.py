import numpy as np

from analyzer.preprocessing import (
    apply_clahe,
    convert_color_space,
    ensure_rgb,
    preprocess_image,
    remove_hair_artifacts,
    resize_image,
)


def test_preprocessing_shapes_and_colour_handling():
    gray = np.zeros((20, 30), dtype=np.uint8)
    rgb = ensure_rgb(gray)
    assert rgb.shape == (20, 30, 3)
    assert resize_image(rgb, (32, 40)).shape == (32, 40, 3)
    assert convert_color_space(rgb, "HSV").shape == (20, 30, 3)
    assert apply_clahe(rgb).shape == rgb.shape


def test_preprocess_image_preserves_original_and_outputs_expected_views():
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    result = preprocess_image(image, (32, 32), remove_hair=True, use_clahe=True)
    assert result["original"].shape == (16, 24, 3)
    assert result["processed"].shape == (32, 32, 3)
    assert result["hair_mask"].shape == (32, 32)


def test_bgr_conversion_is_explicit():
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    assert ensure_rgb(bgr, "BGR")[0, 0].tolist() == [0, 0, 255]

