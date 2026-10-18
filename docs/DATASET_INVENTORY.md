# Available Datasets for Tool Wear / Machining AI

This is the full inventory of public datasets relevant to our use case,
ordered by **size and suitability for deep learning**. Sources verified
from NASA, Kaggle, HuggingFace, Mendeley, Nature Scientific Data,
Fraunhofer FORDATIS, and direct paper links.

---

## Tier 1 — Large enough for deep learning (≥ 1k labeled windows)

| # | Dataset | Size | Signal types | Material | Labels | Access |
|---|---|---|---|---|---|---|
| 1 | **MATWI (KU Leuven)** | **17.7 GB** | Accel, mic, force, images | CK45 / RVS 304 | VB per measurement | rdr.kuleuven.be |
| 2 | **QIT-CEMC (Nature 2025)** | **~340 GB raw** (68 files × 5M rows) | Vibration (3-axis), sound, force (3-axis), torque, images | Ti6Al4V | VB per cycle | figshare / GitHub |
| 3 | **MU-TCM (Mondragon)** | **~50 GB estimated** | Force (3-axis), vibration (3-axis), AE, internal CNC signals | Cast iron, stainless steel | VB per experiment | eBiltegia / Zenodo |
| 4 | **LUH Multivariate TS (Mendeley)** | **12 GB** (6,418 files) | Force (25 kHz), torque, position control deviation (500 Hz) | Cast iron 600-3/S | VB 0–150 µm per run | Mendeley / HuggingFace |
| 5 | **PHM 2010 Challenge** | **~2–5 GB** (~945 cuts × 50 kHz) | Force (3-axis), vibration (3-axis), AE | Stainless steel (≈ HRC 52) | VB per flute per cut | PHM Society / Kaggle |

---

## Tier 2 — Medium (100–1k labeled samples, useful for fine-tuning)

| # | Dataset | Size | Signal types | Material | Labels | Access |
|---|---|---|---|---|---|---|
| 6 | **Nonastreda (HuggingFace)** | 512 samples | Force signals, spectrograms, microscope images | Various | sharp / worn classes | HuggingFace / Mendeley |
| 7 | **Haas VF-1 (Piecuch 2025)** | 968 cycles | Vibration (25 kHz), current (0.5 kHz), 20 channels | 42CrMo4 steel | VB + cycles-to-failure | Figshare / Kaggle |
| 8 | **Vicomtech turning** | 2,054 segments | Control system, sensors, VB | Turning (various) | Vb per segment | GitHub |
| 9 | **CNC Mill Tool Wear (Michigan)** | 18 experiments | Motor currents (X, Y, Z, spindle), 100 ms sampling | Wax | worn / unworn binary | Kaggle (CC0) |
| 10 | **Fraunhofer spindle vibration** | 2 files | Triaxial accel (1,600 Hz) | Steel | sharp vs worn | FORDATIS |

---

## Tier 3 — Small / benchmark only (≤ 200 samples)

| # | Dataset | Size | Notes |
|---|---|---|---|
| 11 | **NASA Milling** | 167 runs, 6 sensor channels | Already downloaded. Good for MLP, too small for CNN-LSTM scratch training. |
| 12 | **Inconel-625 Turning (Mendeley)** | 27 runs, 382M samples | Turning-specific, huge per-run but few runs. Needs heavy subsampling. |
| 13 | **CNC turning roughness (Kaggle)** | ~few hundred rows | Turning, forces + roughness + VB. Compact. |
| 14 | **AISI 5140 turning (Sensors 2020, PMC7472038)** | 9 runs (L9 Taguchi) | Turning, **measured VB + stylus Ra** + force/AE/temp/current summaries. Small but the only external check with measured Ra. |

---

## What we have locally right now

| Dataset | Status | File |
|---|---|---|
| NASA Milling | ✅ Downloaded (13.8 MB parquet) | `data/raw/nasa_milling.parquet` |
| PHM 2010 (high-res tidy mirror) | ✅ Downloaded (12 MB CSV, 104,675 rows) | `data/raw/phm2010_highres.csv` |
| NUAA orthogonal (uniwear bundle) | ✅ Downloaded (57 MB CSV, for future work) | `data/raw/nuaa_highres.csv` |
| Vicomtech turning (2,054 segments, 13 tools) | ✅ Downloaded (3.3 MB CSV) | `data/raw/vicomtech_turning.csv` |
| Haas VF-1 RUL (968 cycles, 14 tools, 42CrMo4 steel) | ✅ Downloaded (1.8 MB features) | `data/raw/haas_features.csv` |
| AISI 5140 turning (9 runs, measured VB + Ra) | ✅ Transcribed (Table 4 of PMC7472038) | `data/raw/aisi5140_turning.csv` |
| LUH Multivariate (KIT) | ✅ Full downloaded (6,418 files / 10.6 GB; helper `scripts/checks/download_luh_subset.py` can also fetch a lighter every-4th mirror) | `data/raw/luh_milling/{filelist.csv,h5/}` |
| MATWI | ❌ Not downloaded | 17.7 GB |
| QIT-CEMC | ❌ Not downloaded | ~340 GB raw |
| MU-TCM | ❌ Not downloaded | ~50 GB |
| Others | ❌ Not downloaded | Various |

---

## Honest sizing for "biggest possible model"

The real constraint isn't disk space — it's **labeled windows you can
actually train on** after preprocessing:

| Dataset | Raw size | Preprocessed windows | Good for |
|---|---|---|---|
| PHM 2010 | ~3 GB | **~50,000 windows** | CNN-LSTM from scratch |
| LUH Multivariate | 12 GB | **~6,000–20,000 windows** | CNN-LSTM from scratch |
| MU-TCM | ~50 GB | **~10,000–30,000 windows** | CNN-LSTM from scratch |
| QIT-CEMC | ~340 GB | **~100,000+ windows** | Big CNN-LSTM, multimodal |
| MATWI | 17.7 GB | **~50,000 windows** (with images) | CNN-LSTM + image fusion |

For **transfer learning**, the recipe becomes:

1. **Pre-train** on the largest available (PHM 2010 or QIT-CEMC)
2. **Fine-tune** on your specific material dataset (NASA = 146 windows, or your own 20–50 cuts)

---

## Recommended download order

| Priority | Dataset | Size | Time to download (50 Mbps) | Why |
|---|---|---|---|---|
| 1 | PHM 2010 | ~3 GB | ~8 min | Best bang/buck for pre-training. Industry standard benchmark. |
| 2 | LUH Multivariate | 10.6 GB (full) | ~30 min | Bigger than PHM2010, multimodal (force + control signals). **Full data downloaded + wired: 9-tool LOTO RMSE 0.0068 mm, `docs/VALIDATION.md` §16, checkpoint `models/luh_hgbr.joblib` (refit on all 9 tools).** |
| 3 | MU-TCM | ~50 GB | ~2.5 hours | Latest (2025), includes internal CNC signals. |
| 4 | QIT-CEMC | ~340 GB | ~17 hours | Massive, but Ti6Al4V-specific. Only if you need Inconel/Ti transfer. |
| 5 | MATWI | 17.7 GB | ~45 min | Multimodal with images. Good for image+signal fusion. |

For the project report: PHM 2010 is pulled and wired (50k+ windows at
the cut level), and LUH is now wired end-to-end as well (subset
download → featuriser → 9-tool LOTO), so the "public benchmarks"
claim holds with measured numbers.
