"""Presentation-ready Streamlit frontend for live image inference."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from analyzer.pipeline import AnalysisConfig, analyze_lesion, select_best_method
from analyzer.reporting import build_analysis_report, image_bytes, json_bytes


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK = REPO_ROOT / "results" / "benchmark_results.csv"
DEFAULT_MODEL = REPO_ROOT / "model" / "unet_skin_lesion.keras"


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


@st.cache_resource(show_spinner=False)
def load_saved_unet(model_path: str):
    from analyzer.unet_segmentation import load_unet_model

    return load_unet_model(model_path)


@st.cache_data(show_spinner=False)
def read_benchmark(path_text: str) -> tuple[pd.DataFrame | None, dict]:
    path = _resolve_repo_path(path_text)
    if not path.exists():
        return None, {"available": False, "message": f"Benchmark file not found: {path}"}
    try:
        table = pd.read_csv(path)
    except Exception as exc:
        return None, {"available": False, "message": f"Benchmark could not be read: {exc}"}
    required = {"method", "mean_dice", "mean_iou"}
    if not required.issubset(table.columns):
        return None, {"available": False, "message": f"Benchmark is missing columns: {sorted(required - set(table.columns))}"}
    best = select_best_method(path)
    metadata_path = path.with_name("run_metadata.json")
    metadata = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
    return table, {"available": True, "path": str(path), "best": best, "metadata": metadata}


def _method_argument(label: str) -> str | None:
    return {
        "Use measured best": None,
        "Otsu thresholding": "otsu",
        "K-means + morphology": "k-means + morphology",
        "U-Net checkpoint": "u-net",
    }[label]


def _show_metric_cards(result: dict):
    features = result["abcd"]
    columns = st.columns(4)
    columns[0].metric("A · Asymmetry", f"{features['asymmetry']:.3f}")
    columns[1].metric("B · Border irregularity", f"{features['border_irregularity']:.3f}")
    columns[2].metric("C · Colour variation", f"{features['colour_variation']:.3f}")
    columns[3].metric("D · Diameter", f"{features['diameter_pixels']:.0f} px")
    st.caption("These are image-derived descriptors, not medical measurements or a diagnosis.")


def _show_benchmark(table: pd.DataFrame | None, status: dict):
    if table is None:
        st.info(status["message"])
        st.code("python scripts/run_benchmark.py --source official_isic --num-samples all --model-path model/unet_skin_lesion.keras")
        return
    best = status.get("best", "otsu")
    st.success(f"Measured best method: {best}")
    display_columns = [
        column
        for column in [
            "dataset",
            "source",
            "evaluation_split",
            "method",
            "n_samples",
            "mean_dice",
            "std_dice",
            "median_dice",
            "mean_iou",
            "std_iou",
            "median_iou",
        ]
        if column in table.columns
    ]
    st.dataframe(table[display_columns], use_container_width=True, hide_index=True)
    chart = table.set_index("method")[[column for column in ["mean_dice", "mean_iou"] if column in table.columns]]
    st.bar_chart(chart, y=list(chart.columns))
    metadata = status.get("metadata", {})
    if metadata:
        columns = st.columns(4)
        columns[0].metric("Selected pairs", metadata.get("selected_pairs", "—"))
        columns[1].metric("Train pairs", metadata.get("train_pairs", "—"))
        columns[2].metric("Validation pairs", metadata.get("validation_pairs", "—"))
        columns[3].metric("Test pairs", metadata.get("test_pairs", "—"))


def _show_theory():
    st.subheader("How the image-processing pipeline works")
    st.markdown(
        """
1. **Standardisation:** the uploaded image is converted to RGB and resized to 256×256 so every method receives the same input shape.
2. **Optional enhancement:** hair removal uses black-hat morphology and inpainting; CLAHE improves local contrast in the LAB luminance channel.
3. **Segmentation:** Otsu estimates a foreground threshold, K-means groups pixels in LAB colour space, and U-Net predicts a pixel-wise probability mask.
4. **Cleanup:** opening removes isolated noise, closing fills small gaps, hole filling repairs interiors, and connected-component filtering keeps the most plausible lesion region.
5. **Measurements:** the selected binary mask drives geometry and colour descriptors plus the displayed contour overlay.
"""
    )
    with st.expander("Why the methods differ"):
        st.markdown(
            """
            Otsu and K-means are classical, image-dependent baselines. They do not learn a training distribution and can be affected by illumination, rulers, hairs, and low contrast. U-Net learns spatial context from annotated masks, but its quality depends on the training split, augmentations, checkpoint, and hardware budget.
            """
        )
    with st.expander("ABCD-style measurement disclaimer"):
        st.markdown(
            """
            The A/B/C/D values are educational image descriptors: asymmetry proxy, border geometry, colour variation, and pixel diameter. They are not validated clinical ABCD criteria, do not estimate physical millimetres without calibration, and must not be used for diagnosis or treatment decisions.
            """
        )


st.set_page_config(page_title="Skin Lesion Analyzer", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 2rem;}
    .hero {padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(135deg, #102a43, #1f6f8b); color: white; margin-bottom: 1rem;}
    .hero h1 {margin-bottom: .25rem;}
    .hero p {margin-bottom: 0; opacity: .9;}
    .small-note {color: #52606d; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero"><h1>Skin Lesion Analyzer</h1><p>Educational image-processing workbench for lesion segmentation and mask-derived measurements.</p></div>',
    unsafe_allow_html=True,
)
st.warning("Educational use only. This application does not diagnose skin disease or replace a qualified clinician.")

with st.sidebar:
    st.header("Experiment settings")
    benchmark_text = st.text_input("Benchmark CSV", value=str(DEFAULT_BENCHMARK.relative_to(REPO_ROOT)))
    model_text = st.text_input("U-Net checkpoint", value=str(DEFAULT_MODEL.relative_to(REPO_ROOT)))
    method_label = st.selectbox("Segmentation method", ["Use measured best", "Otsu thresholding", "K-means + morphology", "U-Net checkpoint"])
    remove_hair = st.checkbox("Remove hair-like artifacts", value=False)
    use_clahe = st.checkbox("Enhance local contrast (CLAHE)", value=False)
    st.caption("Training is intentionally offline. The app only performs inference on uploaded images.")

benchmark, benchmark_status = read_benchmark(benchmark_text)
model_path = _resolve_repo_path(model_text)
status_columns = st.columns(3)
status_columns[0].metric("Benchmark", "Ready" if benchmark_status["available"] else "Missing")
status_columns[1].metric("Best method", benchmark_status.get("best", "—"))
status_columns[2].metric("U-Net", "Available" if model_path.exists() else "Not trained")

analysis_tab, benchmark_tab, theory_tab = st.tabs(["Analyze image", "Benchmark results", "How it works"])

with analysis_tab:
    st.subheader("Upload a dermoscopic image")
    uploaded = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload an image to compare segmentation methods and generate mask-derived measurements.")
    else:
        try:
            image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
        except Exception as exc:
            st.error(f"The uploaded file could not be decoded as an image: {exc}")
        else:
            image_array = np.asarray(image)
            st.image(image, caption=f"Uploaded image · {image.width}×{image.height}", use_container_width=True)
            if st.button("Run analysis", type="primary", use_container_width=True):
                requested_method = _method_argument(method_label)
                if requested_method is None:
                    if not benchmark_status.get("available") or not benchmark_status.get("best"):
                        st.error(
                            "Measured-best analysis requires a valid benchmark CSV. "
                            f"{benchmark_status.get('message', 'Run the benchmark first.')}"
                        )
                        st.stop()
                    selected_best = str(benchmark_status["best"])
                    method_for_analysis = selected_best
                else:
                    selected_best = requested_method
                    method_for_analysis = requested_method
                model = None
                if selected_best in {"u-net", "unet"}:
                    if not model_path.exists():
                        st.error(f"U-Net was selected, but the checkpoint does not exist: {model_path}")
                    else:
                        try:
                            model = load_saved_unet(str(model_path))
                        except Exception as exc:
                            st.error(f"The U-Net checkpoint could not be loaded: {exc}")
                if selected_best not in {"u-net", "unet"} or model is not None:
                    try:
                        with st.spinner("Running preprocessing, segmentation, and feature extraction..."):
                            result = analyze_lesion(
                                image_array,
                                method=method_for_analysis,
                                model=model,
                                config=AnalysisConfig(
                                    benchmark_path=benchmark_text,
                                    model_path=str(model_path) if model is not None else None,
                                    remove_hair=remove_hair,
                                    use_clahe=use_clahe,
                                ),
                            )
                    except (ValueError, OSError, RuntimeError) as exc:
                        st.error(f"Image analysis failed: {exc}")
                        st.stop()
                    if result.get("warning"):
                        st.info(result["warning"])
                    st.success(f"Method used: {result['selected_method']}")
                    view_columns = st.columns(4)
                    view_columns[0].image(result["original"], caption="Original", use_container_width=True)
                    view_columns[1].image(result["preprocessed"], caption="Preprocessed", use_container_width=True)
                    view_columns[2].image(result["mask"] * 255, caption="Predicted mask", use_container_width=True)
                    view_columns[3].image(result["contour_overlay"], caption="Contour overlay", use_container_width=True)
                    st.subheader("ABCD-style image descriptors")
                    _show_metric_cards(result)
                    with st.expander("Processing metadata"):
                        st.json(result["metadata"])
                    download_columns = st.columns(3)
                    download_columns[0].download_button("Download mask", image_bytes(result["mask"], mode="L"), "predicted_mask.png", "image/png", use_container_width=True)
                    download_columns[1].download_button("Download overlay", image_bytes(result["contour_overlay"]), "contour_overlay.png", "image/png", use_container_width=True)
                    download_columns[2].download_button("Download JSON report", json_bytes(build_analysis_report(result)), "analysis_report.json", "application/json", use_container_width=True)

with benchmark_tab:
    st.subheader("Measured method comparison")
    _show_benchmark(benchmark, benchmark_status)
    st.caption("The ranking is read from the executed benchmark CSV; no performance value is hard-coded in the frontend.")

with theory_tab:
    _show_theory()
