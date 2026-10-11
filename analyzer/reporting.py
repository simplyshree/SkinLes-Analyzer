"""Serializable reports and presentation artifacts for the analyzer."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PIPELINE_VERSION = "1.1.0"


def _json_value(value: Any) -> Any:
    """Convert NumPy/Pillow/path values into JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def build_analysis_report(result: dict[str, Any]) -> dict[str, Any]:
    """Return the stable, user-downloadable portion of an analysis result."""
    metadata = result.get("metadata", {})
    report = {
        "report_version": "1.0",
        "pipeline_version": PIPELINE_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
        "selected_method": result.get("selected_method"),
        "warning": result.get("warning"),
        "abcd_style_measurements": result.get("abcd", {}),
        "processing_steps": result.get("processing_steps", []),
    }
    return _json_value(report)


def json_bytes(value: Any) -> bytes:
    return json.dumps(_json_value(value), indent=2, sort_keys=True).encode("utf-8")


def image_bytes(image: np.ndarray, mode: str = "RGB", format_name: str = "PNG") -> bytes:
    """Encode an RGB or single-channel NumPy image for a download button."""
    array = np.asarray(image)
    if mode == "L":
        array = ((array > 0).astype(np.uint8) * 255)
    else:
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        array = np.clip(array, 0, 255).astype(np.uint8)
    import io

    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format=format_name)
    return stream.getvalue()
