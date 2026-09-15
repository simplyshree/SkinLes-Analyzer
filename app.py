"""Streamlit user interface for educational lesion image analysis."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

from analyzer.pipeline import AnalysisConfig, analyze_lesion, select_best_method


st.set_page_config(page_title="Skin Lesion Analyzer", page_icon="🔬", layout="wide")

st.title("Skin Lesion Analyzer")
st.subheader("Computer Vision-Based Lesion Segmentation and ABCD Feature Analysis")
st.warning("This application is an educational image-analysis project and is not intended for medical diagnosis or clinical decision-making.")

with st.sidebar:
    st.header("Analysis settings")
    benchmark_path = st.text_input("Benchmark CSV", value="results/benchmark_results.csv")
    method = st.selectbox("Segmentation method", ["Use measured best", "Otsu", "K-means + morphology", "U-Net"])
    remove_hair = st.checkbox("Apply optional hair/artifact removal", value=False)
    use_clahe = st.checkbox("Apply local contrast enhancement", value=False)
    st.caption("Diameter is reported in pixels because no physical calibration is assumed.")

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(BytesIO(uploaded.getvalue())).convert("RGB")
    image_array = np.asarray(image)
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Analyze Image", type="primary"):
        chosen = None if method == "Use measured best" else method.lower().replace(" + morphology", "")
        with st.spinner("Running preprocessing, segmentation, and image-feature extraction..."):
            result = analyze_lesion(
                image_array,
                method=chosen,
                config=AnalysisConfig(benchmark_path=benchmark_path, remove_hair=remove_hair, use_clahe=use_clahe),
            )
        if result.get("warning"):
            st.info(result["warning"])
        st.caption(f"Method used for this request: `{result['selected_method']}`")
        columns = st.columns(4)
        columns[0].image(result["original"], caption="Original Image", use_container_width=True)
        columns[1].image(result["mask"] * 255, caption="Predicted Segmentation Mask", use_container_width=True)
        columns[2].image(result["contour_overlay"], caption="Contour Overlay", use_container_width=True)
        columns[3].image(result["preprocessed"], caption="Processed Image", use_container_width=True)

        st.subheader("ABCD-style image measurements")
        metric_columns = st.columns(4)
        metric_columns[0].metric("A — Asymmetry", f"{result['asymmetry']:.3f}")
        metric_columns[1].metric("B — Border irregularity", f"{result['border_irregularity']:.3f}")
        metric_columns[2].metric("C — Colour variation", f"{result['colour_variation']:.3f}")
        metric_columns[3].metric("D — Diameter", f"{result['diameter_pixels']:.0f} px")
        st.write("Dominant lesion colours (RGB):", result["dominant_colours"])

        with st.expander("View Image Processing Steps"):
            st.write("  ↓  ".join(result["processing_steps"]))

st.divider()
st.header("Method comparison")
st.write("The table below is read from `results/benchmark_results.csv`; it is not hard-coded. Values are only meaningful when the CSV records an executed, matched evaluation set.")
try:
    import pandas as pd

    benchmark = pd.read_csv(AnalysisConfig().benchmark_path)
    st.dataframe(benchmark, use_container_width=True, hide_index=True)
    st.caption(f"Current measured selection: `{select_best_method(benchmark_path)}`")
except Exception as exc:
    st.info(f"No readable benchmark table is available yet: {exc}")

with st.expander("About this project"):
    st.markdown("""
This project compares Otsu thresholding, K-means plus morphology, and a compact U-Net for lesion-boundary segmentation on ISIC 2018 Task 1 data. The A/B/C/D section reports geometric and colour descriptors derived from the selected mask. It does not classify a lesion and should not be used for diagnosis.

The full reproducible experiment is in `notebooks/skin_lesion_experiments.ipynb`. Dataset archives are downloaded into a cache and are not committed to the repository.
""")

