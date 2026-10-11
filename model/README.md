# Model artifacts

The trained U-Net checkpoint is intentionally not committed by default. Train it offline with the script or notebook and save it as:

```text
model/unet_skin_lesion.keras
```

```powershell
python scripts\train_unet.py --source official_isic --num-samples all --epochs 25
```

The Streamlit app detects whether the artifact exists. A missing artifact is reported clearly and never represented as a trained neural prediction. Training metadata and history are saved under `results/`. For deployment, use a versioned model release or another reproducible artifact store instead of committing a large binary without provenance.

