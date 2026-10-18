# Hybrid AI Expert System for Automated Manufacturing

## 1. What this is

A modular hybrid-AI pipeline that, given a part specification, plans the
manufacturing process, picks a cutting tool, predicts how the tool will
wear, decides whether the cut is safe, and finds the best speed/feed/depth
settings — all through one integrated loop, and all explainable.

**Course:** ME6324 — Expert Systems, Neural Networks, Fuzzy Logic and
Genetic Algorithms in one CAPP + optimisation + diagnosis workflow.

**Live site:** http://127.0.0.1:8502 (FastAPI + single-page dark UI,
clickable 3D Pareto front, full-report export).

## 2. The problem it fixes

Machining a part is a chain of choices nobody makes end-to-end:

- A **CAPP sheet** (which operations, in what order, and the setup & fixturing plan: workholding, datums, clamping) is written by hand
  from a handbook.
- A **cutting tool** is picked from a catalogue, usually the first insert
  that fits the material — not the one that survives the cut.
- **Flank wear (V_B)** and **surface roughness (R_a)** are predicted from
  experience, not from a model.
- **Material removal rate (MRR), tool life and surface finish conflict.**
  Running faster wears the tool out; running slower ruins the finish.
  There is no single best setting — there is a *front* of trade-offs.
- When a tool fails mid-run, nobody re-plans; the part is scrapped.

Each of these is solvable alone with one technique, and each technique
fails alone:

| Approach | Fails because |
| --- | --- |
| Static handbook lookup | Cannot react to real-time tool degradation |
| Pure ML model | Cannot infer operation sequences, justify a tool choice, or explain chatter/chipping |
| Real machining | V_B and R_a evolve nonlinearly; MRR fights tool life and finish |

So the system fuses **symbolic reasoning, learned prediction, uncertainty
handling and Pareto optimisation** in one loop.

## 3. The four stages

```
Part Specs -> CLIPS (symbolic) -> ANN (prediction) -> Fuzzy (state)
              -> NSGA-II (Pareto) -> Dashboard
                      ^                                        |
                      |________________________________________|
                        feedback loop: re-diagnose, re-plan
```

**Stage 1 — CAPP and tool selection (CLIPS).** A real CLIPS 6.4 rule
engine (via `clipspy`) reads an ISO 13399-style tooling catalogue served
from SQLite. Reject rules drop inserts that cannot cut the material or
machine; an eligibility rule ranks what is left. The output is a CAPP
sheet: operations, a setup & fixturing plan (workholding, datums, clamping,
sequence), the envelope of allowed speed/feed/depth, and the chosen insert.

**Stage 2 — Prediction (neural networks).** Five models predict flank wear
and roughness from speed, feed, depth and the tool's properties:
MLP, CNN-BiLSTM and HistGradientBoosting (production tier), plus CNN1D and
CNN-LSTM (reference tier — trained, but overfit, so excluded from
decisions). A hard gate rejects points the models disagree on before the
next stage ever sees them.

**Stage 3 — Diagnosis (fuzzy logic, `scikit-fuzzy`).** Each signal votes
for *fresh*, *caution* or *critical* across material-specific bands; the
part's state is the worst vote. Output: `safe` / `caution` / `critical`,
three membership values, and a one-sentence explanation naming the
driving signal — e.g. *"vibration RMS 1.85 g in caution band."* A
Mamdani Tool Health Index is also produced for the gauge.

**Stage 4 — Optimisation (NSGA-II, `pymoo`).** Searches the envelope for
non-dominated settings trading off MRR, tool life and roughness. The
fuzzy gate makes every returned point safe-by-construction, so the front
is not just optimal — it is *safe*.

## 4. Why "hybrid" matters

- **Symbolic CLIPS** gives a CAPP sheet and a tool choice that a human can
  read and challenge; the rules are inspectable, not a black box.
- **Neural networks** capture the nonlinear physics of wear that no rule
  can encode.
- **Fuzzy logic** turns continuous predictions into a discrete, honest
  state with a sentence anyone can act on — and it refuses to invent a
  verdict when a signal is genuinely out of range.
- **NSGA-II** turns "make it faster, but not too fast" from a vague goal
  into a set of equally-good trade-offs the operator can pick from.

## 5. Honest behaviour: what the system refuses to fake

This is the part that matters most, and it is the result of several
rounds of stress testing.

**The gate is a search constraint, not a filter.** The optimiser only
returns points that pass the fuzzy gate. Random sampling of the same
envelope finds 1/288 = 0.3% safe; NSGA-II finds 24/24 = 100%. The gate
turns a needle-in-a-haystack search into a safe-by-construction front.

**An empty front is explained, not papered over.** If no safe setting
exists, the API returns `no_safe_front` with one of two reasons:

- `no_safe_setting_exists` — no point in the envelope is safe. A real
  case: a 0.3 µm finish asked of an insert whose minimum feed
  (0.15 mm/rev at 0.8 mm nose) already leaves ~0.7 µm of kinematic
  cusp height, so no feed in the envelope can pass the roughness gate.
- `search_missed_safe_point` — a safe point exists (typically a thin
  corner of the envelope) but the search missed it.

**The state classifier is monotone.** An early version used a rule that
needed the critical membership to reach 0.3 before it would declare
"critical". That left a hidden "safe" dead zone just above every mid
threshold — wear 0.30-0.345 mm, vibration 1.5-1.71 g, roughness from
target to 1.15x target. The state curve along one signal went
safe -> caution -> safe -> critical, which is nonsense. A stress test
caught it: 29/300 parts diagnosed *better* at the aggressive corner than
at the midpoint. It is now fixed (max-membership per signal, never
released) and the anomaly rate is 0/300.

**A re-plan that cannot fix a part says so.** `replan` reports
`resolved` with a `reading` of `safe` / `improved` / `no_improvement` /
`no_alternative` — and never invents a fix. If the best alternative is
still dangerous, it says "improved, not fully safe" rather than claiming
victory.

**The physics is calibrated against reality, not demos.** Round 15
re-anchored the generator coefficients, the nose-radius roughness model
and the vibration proxy to measured NASA vibration statistics, and
NSGA-II life now comes from the extended Taylor equation (feed moves
predicted life honestly: ~50 min at the slow P-band end, ~7 min at
300 m/min). Aluminium's old "infeasible everywhere" behaviour was a
calibration failure and is gone — but where a requirement truly cannot
be met (the 0.3 µm case above), the empty front stays and the API says
exactly why.

## 6. Verification (all numbers re-measured this round)

| Check | Result |
| --- | --- |
| `pytest` | 178 passed in 32 s |
| `check_web.py` | 97 endpoint calls, 0 failures |
| `validate_pipeline.py` | 0 failures, 1,876 invariants |
| `stress_test.py 300` | 2,565 invariants, 0 failures, **0 anomalies** in 63 s |
| Coverage | 300/300 parts matched an insert |
| Diagnosis distribution (midpoint) | 15.3% safe / 32.0% caution / 52.7% critical |
| Empty fronts | 55/300 (54 `no_safe_setting_exists`, 1 `search_missed_safe_point`) |
| Dominated points / unsafe in safe-only fronts | 0 / 0 |
| Trusted-model agreement | 45/48 within 0.15 mm |
| Optimiser stability (40-part sample) | 40 stable / 3 check / 0 unstable |

Latency: rules ~8 ms, predict ~0.4 ms, NSGA-II 24x12 ~107 ms, full
pipeline ~115 ms.

## 7. The two dashboards

| Dashboard | URL | Notes |
| --- | --- | --- |
| **New web UI** (FastAPI + single page, dark theme) | http://127.0.0.1:8502 | Clickable Pareto points, full-report export, Big PHM model row, deterministic pipeline runs |
| Streamlit (classic) | http://127.0.0.1:8501 | Original tabbed UI |

Both share the same pipeline; the web UI is the prominent interface.

## 8. How to run

```powershell
cd "C:\Users\AISHWARYA\Downloads\me project"
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# tests
.\.venv\Scripts\python.exe -m pytest -q

# web UI
.\.venv\Scripts\python.exe -m uvicorn hybrid_ai.web.api:app --port 8502

# demo: three parts through all four stages -> interactive 3D Pareto plot
.\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo --smoke
```

## 9. Repository layout

```
me project/
|-- README.md / AGENTS.md / pyproject.toml / requirements.txt
|-- docs/                 # ARCHITECTURE, PROJECT, STORY, VALIDATION, TOOLS, DATASET_INVENTORY
|-- src/hybrid_ai/        # core/ stages/ tools/ web/ dashboard/
|-- tests/                # mirrors src (173 tests)
|-- scripts/training|checks/
|-- archive/agentic-setup/  # archived LLM loop (see its README)
|-- data/                 # raw/ + processed/ (gitignored)
|-- models/               # trained .pt/.joblib checkpoints (LFS)
|-- demo_outputs/         # regenerated demo artifacts (gitignored)
```

## 10. Data and models

### 10.1 Datasets downloaded locally

| Dataset | File in `data/raw/` | Size | Inventory tier |
| --- | --- | --- | --- |
| PHM 2010 | `phm2010_highres.csv` | 12 MB | Tier 1 (large) |
| NASA Milling | `nasa_milling.parquet` | 13.8 MB | Tier 3 (small) |
| NUAA orthogonal | `nuaa_highres.csv` | 57 MB | bundle |
| Vicomtech turning | `vicomtech_turning.csv` | 3.3 MB | **Tier 2** |
| Haas VF-1 | `haas_features.csv` + `haas_metadata.xlsx` | 1.8 MB | **Tier 2** |
| AISI 5140 turning | `aisi5140_turning.csv` | 1 KB | external check (measured VB + Ra) |
| LUH/KIT milling | `luh_milling/` (`filelist.csv` + `h5/`, full 6,418 runs) | 10.6 GB | Tier 1 (large) |

Each has a loader in `src/hybrid_ai/core/` (`phm_loader`, `nasa_loader`,
`unified_loader` = PHM + NUAA, `vicomtech_loader`, `haas_loader`,
`aisi5140_loader`, `luh_loader`), and
the processed caches (`phm_windows.npz`, `nasa_features.npz`,
`unified_windows.npz`, `luh_features.npz`) sit in `data/processed/`.
The LUH h5 data is fetched by `scripts/checks/download_luh_subset.py`
(pass `every=1` for the full set);
its 9-tool leave-one-tool-out artefact is `docs/luh_loto.json`.

### 10.2 Trained models

| Checkpoint | Size | In the app? |
| --- | --- | --- |
| `nasa_mlp.pt` | 24 KB | yes - production wear/roughness model |
| `nasa_cnn1d.pt` | 73.8 MB | yes - reference tier (overfit, not a decision input) |
| `nasa_cnn_lstm.pt` | 302 KB | yes - reference tier |
| `phm_cnn_bilstm.pt` | 4.0 MB | yes - Big PHM model row / uncertainty demo |
| `vicomtech_hgbr.joblib` | 2.8 MB | yes - turning-native production model |
| `luh_hgbr.joblib` | 0.65 MB | yes - milling-native model (LUH signals + log contact time; 9-tool LOTO 0.0068 mm) |
| `unified_cnn_bilstm.pt` | 4.0 MB | no - trained by `train_unified.py`, not wired in |
| `haas_rul_gru.pt` | 200 KB | no - RUL (cycles left), not V_B, so off the diagnose path |

### 10.3 Tier-2 status, honestly

Tier 2 in `docs/DATASET_INVENTORY.md` is five medium datasets
(100-1k labelled samples, meant for fine-tuning). Two of the five have
working code:

| # | Tier-2 dataset | Downloaded | Loader | Model | Wired into the app |
| --- | --- | --- | --- | --- | --- |
| 8 | Vicomtech turning (2,054 segments) | yes | yes | HGBR | **yes** - the turning production model |
| 7 | Haas VF-1 (968 cycles) | yes | yes | GRU RUL | no - trained only; predicts cycles-to-failure, not V_B |
| 6 | Nonastreda (512 samples) | no | no | no | no |
| 9 | CNC Mill Tool Wear (Michigan, wax) | no | no | no | no |
| 10 | Fraunhofer spindle vibration | no | no | no | no |

So: 1 of 5 tier-2 datasets is live, 1 is trained but off-path (RUL, not
wear), and 3 are inventory entries only - no loader, no model, not
downloaded. Separately, the PHM + NUAA unified pool is built and a
CNN-BiLSTM is trained on it (`train_unified.py`), but that model is not
wired in either. Full inventory: `docs/DATASET_INVENTORY.md`.

## 11. Deck claim-by-claim

Every claim on `docs/Hybrid_AI_Manufacturing.pdf` (6 slides) checked
against the code. Everything is implemented except the four items the
deck itself labels *Future Scope*. Three caveats follow the table.

| Slide | Deck claim | Status |
| --- | --- | --- |
| 1 | rule-based + soft-computing pipeline; CAPP, tool selection, multi-objective optimisation | yes - all three live; the CAPP sheet carries the setup & fixturing plan (workholding, datums, clamping, sequence) |
| 2 | wear prediction + fuzzy monitoring flags bad tooling; cuts cycle time, scrap, tool cost | yes - fuzzy gate + replan |
| 3 | NASA Milling: 167 runs, 6 sensor channels (current/vibration/AE), VB labels, trains the ANN | yes - 167 rows, 6 sensor columns, `nasa_mlp.pt` |
| 3 | ISO 13399 catalog: shapes CNMG/WNMG/APMT, nose radius, rake angle, groups P/M/K/N/S | yes - 50 inserts, 10 per group, all fields |
| 3 | FFT/RMS -> normalised feature vectors; catalog -> SQLite (ISO-keyed) | yes - `scipy.fft` in `nasa_loader`, `tool_catalog.sqlite` |
| 4 | four stages CLIPS -> ANN -> Fuzzy -> NSGA-II + dashboard + feedback loop to CLIPS | yes - incl. `replan(stage="clips")` |
| 5 | Streamlit: CAPP sheet, 3D Pareto front (Plotly), diagnostic explanations | yes - `px.scatter_3d` in both frontends |
| 5 | clipspy, PyTorch / scikit-learn, scikit-fuzzy, DEAP / pymoo | yes - all pinned (pymoo used; the deck allows either) |
| 5 | SQLite + SQLAlchemy; 5 team members; single dashboard | yes - plus a FastAPI UI sharing the same pipeline |
| 6 | ANN: low RMSE / MAE on held-out NASA runs (V_B, R_a) | yes - leave-one-run-out numbers in `docs/heldout_nasa.json` |
| 6 | Fuzzy: safe / caution / critical boundary accuracy | yes - boundary tests plus the monotonicity fix (29 -> 0 anomalies) |
| 6 | NSGA-II: Pareto hypervolume and spread | yes - both implemented and checked |
| 6 | CLIPS: rule coverage and correctness vs handbook ground truth | yes - handbook band checks (`TYPICAL_VC`, `VC_CEIL`) |
| 6 | System Speed: planning-time reduction vs manual lookup | yes - measured ~115 ms; see caveat 3 |
| 6 | Physics Validation: outputs vs Taylor tool-life physics | yes - `stages/nsga2/taylor.py`, `taylor_check` |
| 6 | Future Scope: telemetry (OPC-UA/MTConnect), edge compute, closed-loop adaptive control, expansion to Inconel/titanium/multi-axis | no - not built; the slide itself lists these as future |

### Caveats

1. **"Sandvik Coromant" catalog.** The shipped CSV is an ISO 13399-*style*
   illustrative set (50 inserts); its own header states it is *not* a
   proprietary Sandvik Coromant export. Format and fields match the deck;
   the actual Sandvik data does not ship.
2. **NASA "key fields: cutting speed".** The 167-run file carries `feed`,
   `DOC`, `material` and the six sensor channels, but no speed column -
   speed is derived, not raw, so the deck's field list is one item optimistic.
3. **"Low RMSE" and "planning-time reduction".** Held-out V_B RMSE is
   0.250 mm against a train-mean baseline of 0.260 mm - a real but
   marginal gain (R_a is solid at 0.11-0.20 um). And the speed claim has
   no measured manual baseline; it is our own latency only.
