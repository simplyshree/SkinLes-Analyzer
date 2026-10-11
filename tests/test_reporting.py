import json

import numpy as np

from analyzer.pipeline import analyze_lesion
from analyzer.reporting import build_analysis_report, image_bytes, json_bytes


def test_analysis_report_is_json_safe_and_contains_metadata():
    image = np.full((24, 32, 3), 180, dtype=np.uint8)
    result = analyze_lesion(image, method="otsu")
    report = build_analysis_report(result)
    parsed = json.loads(json_bytes(report))
    assert parsed["metadata"]["input_shape"] == [24, 32, 3]
    assert parsed["selected_method"] == "otsu"


def test_image_download_bytes_are_valid_png():
    image = np.zeros((10, 12), dtype=np.uint8)
    encoded = image_bytes(image, mode="L")
    assert encoded[:8] == b"\x89PNG\r\n\x1a\n"
