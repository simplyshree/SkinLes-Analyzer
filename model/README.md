# Model artifacts

The trained U-Net checkpoint is intentionally not committed by default. Train it from the notebook and save it as:

```text
model/unet_skin_lesion.keras
```

The Streamlit app detects whether the artifact exists. A missing artifact is reported clearly and never represented as a trained neural prediction. For deployment, use a versioned model release or another reproducible artifact store instead of committing a very large binary without provenance.

