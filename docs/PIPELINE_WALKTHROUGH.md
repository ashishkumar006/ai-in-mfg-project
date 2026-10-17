# Pipeline Walkthrough

A plain-English explanation of what this system currently does, what it
needs before it becomes production-grade, and what the planned deep-
learning upgrades will look like. No implementation is included in
this document — it is a roadmap, not a todo list.

---

## 1. The current system — what works today

The codebase already ships a fully wired four-stage pipeline. You can
run it right now with `python -m hybrid_ai.dashboard.demo` and the
full flow completes end-to-end in about 20 seconds:

```
PartSpec  →  CLIPS (rules)  →  ANN (MLP)  →  Fuzzy  →  NSGA-II  →  Dashboard
```

**What each stage uses today:**

- **CLIPS module** (`src/hybrid_ai/stages/clips/rules.py`) — pure Python rule
  predicates: ISO match, operation match, steel-finishing, heavy-
  roughing, superalloy-cautious. Selects from 15 hardcoded `ToolCandidate`
  entries. No CLIPS engine is required; the rule interface mirrors CLIPS
  syntax so a future swap to `clipsPy` is a one-file change.

- **ANN module** (`src/hybrid_ai/stages/ann/model.py`) — PyTorch 2-layer MLP,
  `(32, 32)` hidden, Adam optimiser, MSE loss, 60 epochs on 1 200
  synthetic rows. Input is an 11-element vector: V_c, feed, ap, nose
  radius, rake angle, 5-dim ISO one-hot, hardness. Outputs: V_B (mm)
  and R_a (µm), clipped to physical ranges.

- **Fuzzy module** (`src/hybrid_ai/stages/fuzzy/diagnoser.py`) — Mamdani-style
  deterministic classifier using triangular and trapezoidal membership
  functions. Three signals: predicted V_B, predicted R_a, vibration
  RMS. Maps to `safe / caution / critical` with a textual explanation.

- **NSGA-II module** (`src/hybrid_ai/stages/nsga2/optimiser.py`) — pymoo
  NSGA-II searching V_c, feed, ap inside the tool's envelope. Objectives:
  maximise MRR, maximise tool life (calibrated `6.0 / V_B`, ~20 min to criterion), minimise
  R_a. Returns a non-dominated Pareto front. Grid-sweep fallback if pymoo
  is absent.

- **Dashboard** (`src/hybrid_ai/dashboard/app.py`) — four Streamlit tabs:
  CAPP (operation plan, candidates, selected tool, envelope, notes),
  Pareto (3D Plotly, top-5 tables), Diagnosis (state, V_B, R_a,
  membership bars, explanation), Pipeline trace (tabular summary).

- **Data layer** — everything is synthetic. `core/synthetic.py` holds
  15 hardcoded inserts and a `generate_synthetic_dataset` that produces
  physically plausible training rows from a Taylor-style formula. The
  SQLite repository (`core/sqlite_loader.py`) exists and is wired to the
  catalog but currently only serves the in-memory fixture.

---

## 2. What is missing — the gaps

The pipeline runs and the demo is convincing, but the system is not
yet usable on real industrial data. The gaps are:

1. **No real NASA data ingestion.** The ANN trains on synthetic rows.
   The NASA Milling dataset (167 runs, 6-channel sensor data, V_B) is
   publicly available but not downloaded or parsed.

2. **No proprietary Sandvik export.** The catalog now *is* parsed:
   `core/tooling_catalog.py` reads the bundled ISO 13399-representation
   CSV (`src/hybrid_ai/assets/tool_catalog_iso13399.csv`, 50 inserts
   from publicly documented families — illustrative, not Coromant's
   database) into the ISO-keyed SQLite store at
   `data/processed/tool_catalog.sqlite`. Every (group, operation) cell
   has 2 inserts. Swapping in a true vendor export only requires a
   conforming CSV.

3. **No raw-signal path.** The ANN only accepts engineered features
   (V_c, feed, ap, …). It cannot read raw vibration / acoustic
   emission / current time-series. That blocks the use of
   PHM 2010 and MATWI-scale datasets.

4. **No deep-learning models beyond MLP.** A 2×32 MLP is fine for a
   prototype but caps out quickly on real data. CNN-1D and CNN-LSTM
   architectures are absent.

5. **No transfer-learning pathway.** Fine-tuning a model trained on
   public benchmarks (PHM2010) to your specific material (Inconel,
   titanium, etc.) would dramatically improve accuracy with few local
   samples — but this workflow does not exist.

6. **No physics-informed training.** The loss is pure MSE. Adding a
   Taylor tool-life residual term would anchor predictions to known
   physics and improve extrapolation outside the training envelope.

7. **No confidence intervals.** The `model_confidence` field in
   `WearPrediction` is a hand-tuned heuristic, not a principled
   uncertainty estimate.

---

## 3. The phases — what needs to be done, in order

### Phase 1 — Wire the real datasets (2–3 hours of code)

The goal is to replace `generate_synthetic_dataset` with a real loader
without changing any downstream code.

1.1 **Download NASA Milling dataset**

- Source: `https://data.nasa.gov/dataset/milling-wear`
- 167 rows, CSV, columns: case, run, VB, time, DOC, feed, material,
  AE1–AE6, Vibe1–Vibe6, SM1–SM3
- Place at `data/raw/nasa_milling.csv`

1.2 **Write `core/nasa_loader.py`**

- Read the CSV.
- Parse the six sensor channels and compute FFT / RMS aggregates so
  the feature vector matches what `predict_for_point` expects
  (V_c, feed, ap, nose_radius, rake_angle, ISO one-hot, hardness).
- For the 167 runs, produce `(X, y)` where `X` is `(167, 11)` and
  `y` is `(167, 2)`.
- Add `core/windower.py` later for raw-signal windows (Phase 2).

1.3 **Write `core/catalog_parser.py`**

- Ingest a JSON / CSV export from the Sandvik tool configurator.
- Upsert into the existing SQLite repository via
  `ToolCatalogRepository.upsert`.
- CLIPS already queries SQLite — no CLIPS changes needed.

1.4 **Add a data-source toggle in the dashboard**

- In `dashboard/app.py`'s `get_model()`, check for `data/raw/nasa_milling.csv`.
- If present, load from disk; otherwise fall back to synthetic.
- Same for the catalog (already the case): `get_catalog()` parses the
  bundled ISO 13399 CSV every process start, replaces the SQLite rows
  (`data/processed/tool_catalog.sqlite`), and falls back to the 15-insert
  built-in list only if the asset cannot be parsed.

### Phase 2 — Add deep-learning models (4–6 hours of code)

The goal is a model registry so the dashboard can pick MLP, 1D-CNN,
or CNN-LSTM without touching CLIPS / Fuzzy / NSGA-II.

2.1 **Define the base API**

```python
class BaseModel(ABC):
    def fit(self, X, y, config): ...
    def predict(self, X): ...
    def evaluate(self, X, y): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```

`WearRoughnessModel` (the MLP) already implements this shape; refactor
it to inherit from `BaseModel`.

2.2 **Implement `CNN1DModel`**

- Input: `(batch, n_channels, window_length)`
- Layers:
  - `Conv1D(32, kernel=5, ReLU)` → `MaxPool1D(2)`
  - `Conv1D(64, kernel=3, ReLU)` → `MaxPool1D(2)`
  - `Flatten` → `Linear(64)` → `ReLU` → `Dropout(0.2)` → `Linear(2)`
- Trains in ~5–10 min on CPU with 50k windows.

2.3 **Implement `CNNLSTMModel`**

- Same CNN front-end → `LSTM(64, num_layers=2, batch_first=True)`
- Take the last hidden state → `Linear(2)`
- Trains in ~15–30 min on CPU; ~2 min on GPU.

2.4 **Write `core/windower.py`**

- Slice raw multi-channel time-series into fixed-length windows
  (e.g. 2 000 samples) with stride.
- Handle formats: NASA CSV, PHM2010 `.mat`, MATWI `.csv`.
- Label each window with the corresponding V_B or R_a measurement.
- Output: `(n_windows, n_channels, window_len)` float32 tensor.

2.5 **Transfer-learning utility**

- Pre-train `CNNLSTMModel` on PHM2010 (~50k windows, public).
- Save to `models/pretrained_phm2010.pt`.
- Fine-tune last FC layer on NASA (167 runs → ~3k windows) in ~30 s.
- Expose as `ANNModelRegistry.fine_tune(pretrained_path, X, y)`.

### Phase 3 — Dashboard + demo updates (1–2 hours)

3.1 **Model selector in sidebar**

- Radio or dropdown: MLP / 1D-CNN / CNN-LSTM / Transfer-learned.
- Show parameter count, training time, RMSE on held-out data.
- `st.cache_resource` should key off the selected model name so each
  model trains/loads independently.

3.2 **Update `demo.py`**

- Run all available models and print a comparison table.
- Show training time, held-out RMSE/MAE, and a sample prediction
  side by side.

3.3 **Add `models/` directory**

- `.gitignore` it.
- Save / load trained weights using the `save` / `load` methods on
  `BaseModel`.

### Phase 4 — Polish (2–3 hours)

4.1 **Physics-informed loss**

- Add an optional Taylor tool-life residual term to the training loss:
  `L_total = MSE + λ * (log(predicted_life) − log(0.3 / V_B))^2`
- Tune λ as a `TrainConfig` field.

4.2 **Attention visualization (CNN-LSTM only)**

- Extract the LSTM attention weights after a forward pass.
- Plot them in the dashboard as a heatmap over the time window so
  the operator can see *which part of the cut* triggered the
  diagnosis.

4.3 **Confidence intervals**

- Use Monte Carlo dropout at inference time: run `predict` 20 times
  with dropout enabled and report the standard deviation as
  `model_confidence`.
- Replace the current hand-tuned confidence heuristic.

4.4 **Validation report**

- Populate `docs/VALIDATION.md` with held-out metrics for every model
  and every dataset (synthetic, NASA, PHM2010, MATWI).
- Include Pareto hypervolume and spread for NSGA-II.

---

## 4. Data — what is available and what we will use

### Already available (public, free)

| Dataset | Size | Best for | Access |
|---|---|---|---|
| NASA Milling | 167 runs, 14 cols | Prototype validation | data.nasa.gov |
| PHM 2010 Challenge | ~50k windows, 50 kHz | 1D-CNN / CNN-LSTM | PHM Society / Kaggle |
| QIT-CEMC | Medium, vibration+force+images | 1D-CNN with transfer | GitHub + Baidu |
| Mendeley Multivariate TS | 6 418 files, 9 cutters | Cross-machine generalisation | Mendeley Data |

### Large / expensive (use only if you have the bandwidth)

| Dataset | Size | Notes |
|---|---|---|
| MATWI (KU Leuven) | 17.7 GB, 17 tools × 100 measurements | Multimodal (images + accel + mic + force). Preprocessing alone is hours. |
| Inconel-625 Turning | 382 M samples, 27 runs | Good for superalloy fine-tuning, but needs heavy subsampling for CPU training. |

### What we will wire first

1. **NASA Milling** — smallest, easiest, directly replaces the
   synthetic generator in Phase 1.
2. **PHM 2010** — the benchmark for the 1D-CNN and CNN-LSTM in
   Phase 2. Pre-train on this; fine-tune on NASA.
3. **MATWI** — optional Phase 4 stretch goal if the team wants a
   multimodal demo.

---

## 5. The "deep learning" answer — what we will actually build

Starting from the current MLP, the deep-learning upgrade path is:

```
today:  MLP  (engineered features, 11-dim, 20 s train)
  ↓
Phase 2a: CNN1D  (raw sensor windows, ~5-10 min train)
  ↓
Phase 2b: CNN-LSTM  (CNN front-end + LSTM temporal model, ~30 min CPU)
  ↓
Phase 2c: Transfer learning  (pre-train on PHM2010, fine-tune on NASA, ~30 s)
```

The key insight is that **the rest of the pipeline does not change**.
CLIPS, Fuzzy, and NSGA-II all call the same `predict_for_point`
helper, which in turn calls `model.predict`. As long as the new model
classes expose `fit` / `predict` / `evaluate`, they drop straight in.
The dashboard just needs a dropdown to pick which model to use.

### CNN1D architecture (for NASA / PHM2010-scale data)

```
Input: (batch, n_channels=6, window_len=2000)
  → Conv1D(32, kernel=5, padding=2, ReLU)
  → MaxPool1D(2)
  → Conv1D(64, kernel=3, padding=1, ReLU)
  → MaxPool1D(2)
  → Flatten
  → Linear(256, ReLU)
  → Dropout(0.2)
  → Linear(2)   # V_B, R_a
```

~18k parameters. Trains in 5–10 min on CPU with 50k windows.

### CNN-LSTM architecture (for PHM2010 / MATWI-scale data)

```
Input: (batch, n_channels=6, window_len=2000)
  → Conv1D(32, kernel=5, ReLU) → MaxPool1D(2)
  → Conv1D(64, kernel=3, ReLU) → MaxPool1D(2)
  → LSTM(64, num_layers=2, batch_first=True)
  → Attention pooling  (learn weights over time steps)
  → Linear(2)   # V_B, R_a
```

~45k parameters. Trains in 15–30 min on CPU; ~2 min on GPU.

### Transfer-learning workflow

1. Download PHM2010 (public, ~50k windows).
2. Train `CNNLSTMModel` from scratch on PHM2010. Save weights.
3. For a new material (e.g. Inconel), replace the last `Linear(2)`
   layer with a fresh head. Freeze the CNN + LSTM layers.
4. Fine-tune only the new head on the small local dataset (NASA's
   167 runs, or your own 20–50 cuts).
5. Unfreeze all layers and continue training at a lower learning rate
   for 5–10 epochs.

This is the standard recipe in the literature (BRANN on NASA, MCAG on
PHM2010, etc.) and it's what gets you from "research model" to
"actually useful on the shop floor".

---

## 6. The project as a whole — where we are and where we're going

| Layer | Current state | Target state |
|---|---|---|
| CLIPS | Working rule engine in Python | Replace with `clipsPy` or keep Python rules |
| ANN | MLP, synthetic data | CNN1D + CNN-LSTM + transfer learning, real NASA/PHM data |
| Fuzzy | Working Mamdani classifier | Tune thresholds on real sensor data; optionally use `scikit-fuzzy` directly |
| NSGA-II | Working pymoo NSGA-II | Add physics constraints; add robustness analysis (multiple ANN seeds) |
| Data | Synthetic generator + 15 hardcoded inserts | NASA loader + Sandvik catalog parser + windower for raw signals |
| Dashboard | 4 tabs, works on synthetic | Model selector, attention viz, confidence intervals, real data |
| Tests | 29/29 pass | Add integration tests that run the full pipeline on NASA |

The architecture is solid. The implementations are correct. What's
missing is the data plumbing and the model depth. Phases 1 and 2
address exactly those two gaps, and they are mostly additive — new
files, not rewrites.
