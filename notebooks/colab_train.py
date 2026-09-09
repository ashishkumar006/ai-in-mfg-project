"""Colab-ready training script for the Hybrid AI Manufacturing pipeline.

Usage in Google Colab (T4 GPU, free):

    1. Open https://colab.research.google.com
    2. Create a new notebook, set runtime to GPU (T4)
    3. Upload the entire `me project/src` folder (or clone from a repo)
    4. Upload `data/raw/nasa_milling.parquet` (13.8 MB)
    5. Paste this entire file into a cell and run

The script will:
    - Install dependencies
    - Verify GPU is available
    - Train MLP, CNN1D, and CNN-LSTM on NASA
    - Save trained models
    - Print a comparison table

Estimated runtime on T4: ~5 minutes total
"""

# %% [markdown]
# # Hybrid AI Manufacturing — GPU Training
#
# This notebook trains the three ANN architectures on the NASA Milling
# dataset using GPU acceleration. Paste the contents of this file into
# a single Colab cell and run.

# %%
# Install dependencies (Colab has most of these pre-installed)
import subprocess
import sys

def _pip(pkg):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

for pkg in ["torch", "numpy", "scipy", "pandas", "pymoo", "plotly", "streamlit"]:
    try:
        __import__(pkg)
    except ImportError:
        _pip(pkg)

# %%
# Verify GPU
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_mem / 1e9, "GB")
else:
    print("WARNING: No GPU detected. Training will be slow.")

# %%
# Add src/ to path (Colab puts uploaded files in /content/)
import sys
sys.path.insert(0, "/content/src")

# %%
# Import our modules
from hybrid_ai.ann.model import (
    TrainConfig, DLConfig, WearRoughnessModel, WearRoughnessDL,
)
from hybrid_ai.core.nasa_loader import (
    load_nasa_dataset, load_nasa_windows, nasa_feature_dim,
)
import numpy as np
import time

# %% [markdown]
# ## 1. MLP on NASA featurised data
# Fastest to train, works on small datasets. Uses 51 hand-crafted
# features (FFT, RMS, crest factor, spectral centroid per sensor channel).

# %%
X_eng, y_eng = load_nasa_dataset()
print(f"X shape: {X_eng.shape}, y shape: {y_eng.shape}")

t0 = time.time()
mlp = WearRoughnessModel()
mlp.fit(X_eng, y_eng, TrainConfig(epochs=80, hidden=(64, 64), learning_rate=1e-3))
mlp_time = time.time() - t0
mlp_metrics = mlp.evaluate(X_eng, y_eng)
print(f"MLP training: {mlp_time:.2f}s, RMSE={mlp_metrics['rmse']:.4f}, MAE={mlp_metrics['mae']:.4f}")
mlp.save("models/nasa_mlp.pt")

# %% [markdown]
# ## 2. CNN1D on NASA raw sensor windows
# Operates directly on (6 channels x 9000 samples) raw signals.
# No feature engineering — learns filters from scratch.

# %%
X_win, y_win = load_nasa_windows()
print(f"X shape: {X_win.shape}, y shape: {y_win.shape}")

t0 = time.time()
cnn = WearRoughnessDL(config=DLConfig(
    model_type="cnn1d", epochs=40, batch_size=16,
    n_channels=6, window_len=9000, learning_rate=1e-3,
))
cnn.fit(X_win, y_win)
cnn_time = time.time() - t0
cnn_metrics = cnn.evaluate(X_win, y_win)
print(f"CNN1D training: {cnn_time:.2f}s, RMSE={cnn_metrics['rmse']:.4f}, MAE={cnn_metrics['mae']:.4f}")
cnn.save("models/nasa_cnn1d.pt")

# %% [markdown]
# ## 3. CNN-LSTM on NASA raw sensor windows
# Best for temporal patterns. On T4, 40 epochs of 146 windows
# trains in ~30-60 seconds.

# %%
t0 = time.time()
lstm = WearRoughnessDL(config=DLConfig(
    model_type="cnn_lstm", epochs=40, batch_size=16,
    n_channels=6, window_len=9000, lstm_hidden=64, lstm_layers=2,
    learning_rate=1e-3, dropout=0.2,
))
lstm.fit(X_win, y_win)
lstm_time = time.time() - t0
lstm_metrics = lstm.evaluate(X_win, y_win)
print(f"CNN-LSTM training: {lstm_time:.2f}s, RMSE={lstm_metrics['rmse']:.4f}, MAE={lstm_metrics['mae']:.4f}")
lstm.save("models/nasa_cnn_lstm.pt")

# %% [markdown]
# ## 3b. CNN-BiLSTM on PHM 2010 windows (the big model)
# 1M params, 6,496 windows (7 ch x 256 @25 Hz), leave-one-cutter-out:
# train c1+c4, validate on unseen cutter c6. Skipped if the CSV is absent.

# %%
from pathlib import Path as _Path

phm_csv = _Path("data/raw/phm2010_highres.csv")
if phm_csv.exists():
    from hybrid_ai.core.phm_loader import load_phm_windows
    Xtr, ytr, _ = load_phm_windows(window=256, stride=16, split="train")
    Xte, yte, _ = load_phm_windows(window=256, stride=16, split="test")
    print(f"PHM windows: train {Xtr.shape}, test {Xte.shape}")
    t0 = time.time()
    big = WearRoughnessDL(config=DLConfig(
        model_type="cnn_bilstm", epochs=25, batch_size=64,
        n_channels=7, window_len=256, lstm_hidden=128, lstm_layers=2,
        learning_rate=1e-3, dropout=0.2,
    ))
    big.fit(Xtr, ytr)
    big_time = time.time() - t0
    big_metrics = big.evaluate(Xte, yte)
    print(f"CNN-BiLSTM: {big_time:.2f}s, RMSE={big_metrics['rmse']:.4f}, "
          f"MAE={big_metrics['mae']:.4f} (held-out cutter c6)")
    big.save("models/phm_cnn_bilstm.pt")
else:
    print("data/raw/phm2010_highres.csv not found, skipping big-model section.")
    big_time, big_metrics = float("nan"), {"rmse": float("nan"),
                                           "mae": float("nan")}

# %% [markdown]
# ## 4. Comparison table

# %%
import pandas as pd

results = pd.DataFrame([
    {"Model": "MLP (featurised)", "Params": "~5k", "Train time (s)": round(mlp_time, 2),
     "RMSE": round(mlp_metrics['rmse'], 4), "MAE": round(mlp_metrics['mae'], 4)},
    {"Model": "CNN1D (raw windows)", "Params": "~18.4M", "Train time (s)": round(cnn_time, 2),
     "RMSE": round(cnn_metrics['rmse'], 4), "MAE": round(cnn_metrics['mae'], 4)},
    {"Model": "CNN-LSTM (raw windows)", "Params": "~74k", "Train time (s)": round(lstm_time, 2),
     "RMSE": round(lstm_metrics['rmse'], 4), "MAE": round(lstm_metrics['mae'], 4)},
    {"Model": "CNN-BiLSTM (PHM, held-out c6)", "Params": "~1M", "Train time (s)": round(big_time, 2),
     "RMSE": round(big_metrics['rmse'], 4), "MAE": round(big_metrics['mae'], 4)},
])
print(results.to_string(index=False))

# %% [markdown]
# ## 5. Download trained models to your laptop
#
# Run this cell to get the model files. Or use the Colab file browser
# on the left sidebar to download `models/nasa_mlp.pt`,
# `models/nasa_cnn1d.pt`, and `models/nasa_cnn_lstm.pt` manually.

# %%
from google.colab import files
import os

os.makedirs("download", exist_ok=True)
for f in os.listdir("models"):
    src = f"models/{f}"
    if os.path.isfile(src):
        files.download(src)

# %% [markdown]
# ## 6. Next step: LUH Multivariate dataset
#
# To scale up, download the LUH dataset (12 GB parquet) and preprocess:
#
#     from huggingface_hub import hf_hub_download
#     path = hf_hub_download(
#         repo_id="alpha-by-beta/Multivariate_time_series_data_of_milling_processes_with_varying_tool_wear_and_machine_tools",
#         filename="dataset.parquet",
#         repo_type="dataset",
#         local_dir="/content/luh",
#     )
#
# Then preprocess with the windower and train CNN-LSTM on ~6,418 windows.
# Expected: 10-15 minutes on T4, RMSE potentially 0.02-0.05.
