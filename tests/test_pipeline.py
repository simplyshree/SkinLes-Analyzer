import numpy as np
import pytest

from analyzer.pipeline import analyze_lesion, select_best_method


def test_pipeline_returns_required_outputs():
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[18:46, 20:44] = [50, 40, 40]
    result = analyze_lesion(image, method="otsu")
    assert result["mask"].shape == (256, 256)
    assert result["contour_overlay"].shape == (256, 256, 3)
    assert "asymmetry" in result and "diameter_pixels" in result


def test_pipeline_rejects_invalid_image():
    with pytest.raises(ValueError):
        analyze_lesion(np.zeros((4, 4, 2), dtype=np.uint8), method="otsu")


def test_synthetic_smoke_results_do_not_select_a_production_method(tmp_path):
    path = tmp_path / "smoke.csv"
    path.write_text("dataset,method,mean_dice,mean_iou\nsynthetic_smoke_fixture,K-means + Morphology,0.99,0.99\n", encoding="utf-8")
    assert select_best_method(path) == "otsu"
