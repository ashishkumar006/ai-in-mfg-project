# Hybrid AI-Expert System for Automated Manufacturing

A modular, fully-offline decision-support system for CNC machining. Give it a
part, and it answers the three questions every machinist faces — **how to make
it**, **with what tool**, and **at what cutting parameters** — then predicts how
the tool will wear and whether the cut is *safe*, and finally searches for the
best trade-off between output, tool life and finish. Every number it shows is
produced by a pipeline stage and can be traced back to its source.

It is a **hybrid** system: it fuses a rule-based expert system, neural networks,
fuzzy logic, and a multi-objective evolutionary optimiser into one loop, because
no single method handles all four kinds of reasoning well.

- **Symbolic rules** (CLIPS) select the tool and plan the process — inspectable,
  and able to say *why* and to *refuse* outside their knowledge.
- **Neural networks** learn the nonlinear wear behaviour from real machining data.
- **Fuzzy logic** turns predicted numbers into graded verdicts with a reason.
- **NSGA-II** explores the whole trade-off *front* instead of optimising one number.

Everything runs locally: no API keys, no model downloads, deterministic given a seed.

---

## 1. What problem does this solve?

A CNC shop must answer three linked questions for **every** part:

1. **How do I make it?** — the process plan (CAPP sheet): the operations in
   order, plus the **setup & fixturing** plan (workholding, datums, clamping,
   tool holding and the setup sequence).
2. **With what tool?** — the right insert (geometry + material grade) and a safe
   envelope of cutting speed, feed and depth for this workpiece.
3. **At what parameters, and is it safe?** — the specific speed/feed/depth that
   balances removal rate against tool life and surface finish, with a verdict.

Today each answer comes from a different, disconnected place. This project wires
them into one pipeline.

### Why "hybrid"?

| Approach | What it cannot do |
| --- | --- |
| Static handbook lookup | Cannot react to real tool degradation or justify a choice against live data |
| Pure machine learning | Cannot infer operation sequences, justify a tool choice, or explain a chatter/chipping fault |
| Rules alone | Cannot predict the nonlinear wear that develops during the cut |
| Single-objective optimisation | Removes the trade-off the operator actually cares about |

The system therefore combines all four, each doing the part it is good at, and
passes a clean, typed result from stage to stage.

---

## 2. How it works

```
   Part spec
      │
      ▼
┌───────────────┐   insert + envelope    ┌──────────────────┐
│ 1. CLIPS      │ ─────────────────────► │ 2. ANN           │
│    rules      │   + CAPP plan + setup  │    wear model    │
└───────────────┘                        └──────────────────┘
      ▲                                           │  V_B, R_a
      │                                           ▼
      │                                  ┌──────────────────┐
      │                                  │ 3. Fuzzy logic   │
      │   feedback: re-plan /            │    safe/caution/ │
      │   propose recalibration          │    critical      │
      │                                  └──────────────────┘
      │                                           │  verdict
      │                                           ▼
      │                                  ┌──────────────────┐
      └──────────────────────────────────│ 4. NSGA-II       │
         (verdict gates the search;      │    Pareto front  │
          clicking a point re-diagnoses) └──────────────────┘
                                                   │
                                                   ▼
                                   Two front-ends (FastAPI console, Streamlit)
```

**Stage 1 — CAPP & tool selection (symbolic, CLIPS).**
Reads the part spec (ISO material group P/M/K/N/S, operation, machine class,
hardness, target finish). A real CLIPS 6.4 engine (via `clipspy`) runs
hand-authored if-then rules against an ISO 13399-style insert catalogue (SQLite):
reject rules drop inserts that cannot cut the material or operation, an
eligibility rule ranks what survives, and the stage emits an **operation plan**,
a **setup & fixturing plan**, the recommended **envelope** (speed/feed/depth) and
the chosen **insert**. When nothing fits it says why (per-rule survival counts)
and lists the closest inserts.

**Stage 2 — Wear prediction (learned, ANN).**
An MLP predicts **flank wear V_B** (mm) and **surface roughness R_a** (µm) for a
given insert at a given operating point. Outputs carry a Monte-Carlo-dropout
spread so the UI can state its confidence, and are clipped to physically
possible ranges. (See §5 — other datasets train their own models.)

**Stage 3 — Safety diagnosis (fuzzy logic).**
The numbers alone do not say what to do, so fuzzy membership functions map
V_B, R_a and vibration through trapezoids to a graded verdict — **safe**,
**caution** or **critical** — that scales with the part's own finish tolerance,
plus a plain-language explanation. It also produces a Mamdani **Tool Health
Index** (0–100).

**Stage 4 — Multi-objective optimisation (NSGA-II, via `pymoo`).**
There is no single best setting, so this stage evolves a population of candidate
speed/feed/depth combinations and keeps the non-dominated ones: **maximise**
material-removal rate, **maximise** tool life (extended Taylor equation with
per-group handbook constants), **minimise** roughness. The fuzzy verdict is
wired in as a hard constraint, so with **safe-only** on (the default) the
optimiser literally cannot propose a cut the diagnoser calls unsafe. It reports
the exact hypervolume and spacing of the front.

**The loop is closed.** A bad verdict can be fed back to Stage 1 to re-plan
(swap insert, split roughing/finishing) or to propose a recalibration of the
fuzzy limits; clicking any Pareto point re-runs the diagnosis on that exact
setting.

---

## 3. Quickstart

Requires Python 3.11+ (developed on Windows + PowerShell 5.1; the code is
cross-platform).

```powershell
# 1) environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2) run the tests (fast, ~40 s)
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest -q

# 3) run the end-to-end demo (three very different parts through all four stages)
.\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo --smoke
#   -> writes demo_outputs\pareto.html and demo_outputs\SUMMARY.md

# 4) launch a front-end (see §4)
.\.venv\Scripts\streamlit.exe run src\hybrid_ai\dashboard\app.py
```

> The package lives under `src/`, so set `PYTHONPATH=src` (or `cd src`) for
> direct `python -m ...` runs. `pytest` is pre-configured and needs no extra flags.

---

## 4. The two front-ends

Both share the **same** pipeline; they are two views of one engine.

| Front-end | URL | What it is |
| --- | --- | --- |
| **Web console** (FastAPI + a single dark page, `src/hybrid_ai/web/`) | http://127.0.0.1:8502 | Describe a part, press *Run pipeline*, get tool selection + setup plan + diagnosis + a clickable 3D Pareto front. Includes a **lab-condition picker** (PHM 2010, Vicomtech, LUH/KIT) and a one-click full-report export. |
| **Streamlit dashboard** (`src/hybrid_ai/dashboard/app.py`) | http://127.0.0.1:8501 | The original tabbed UI: CAPP sheet, Pareto, diagnosis, the **all-models comparison table**, and a pipeline trace. |

```powershell
# web console (from the repo root, with PYTHONPATH set)
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m uvicorn hybrid_ai.web.api:app --port 8502
```

There is also a **guided tour** that explains the four stages in plain language,
reachable from the console at `/pipeline.html`.

---

## 5. Models — and how well they actually do

The system ships **eight** trained checkpoint files; **six are wired in**. It is
important to be honest about the difference between them: the console's live
prediction uses exactly one model (the MLP), while the dashboard compares them
side by side. Each model is only valid inside the regime it was trained on —
**there is no cross-lab validation**.

| Model | Trained on | Held-out result | Baseline (guess mean) | Role |
| --- | --- | --- | --- | --- |
| **MLP** `nasa_mlp.pt` | 1,200 synthetic rows + ~145 real NASA rows (11 features) | NASA leave-one-run-out RMSE **0.2496 mm** | 0.2596 | **the console's wear model** (a blended generalist, not NASA-only) |
| **PHM CNN-BiLSTM** `phm_cnn_bilstm.pt` | PHM 2010 windows, 7×256 (force/vib/AE) | RMSE **0.0171 mm** — 85% better, held-out cutter `c6` | 0.0444 | dashboard comparison |
| **Vicomtech HGBR** `vicomtech_hgbr.joblib` | Vicomtech turning, 152 features | RMSE **0.0257 mm** — 85% better | 0.0655 | dashboard comparison |
| **LUH HGBR** `luh_hgbr.joblib` | LUH/KIT cast-iron milling, 17 features | RMSE **0.0068 mm** — 95% better (9-tool leave-one-tool-out) | 0.0302 | dashboard comparison |
| **NASA CNN1D / CNN-LSTM** `nasa_cnn1d.pt`, `nasa_cnn_lstm.pt` | NASA raw windows, 6×9000 | RMSE ≈ 2.54 mm — *memorises its training runs* | — | reference/baseline tier (shows why windowed CNNs need more data) |

Two further checkpoints are kept on disk but **not wired** (`unified_cnn_bilstm.pt`,
`haas_rul_gru.pt`); they under-perform and are excluded from the app.

The honest takeaway: the lab models are genuinely strong **on their own sensor-rich
data** (0.007–0.026 mm), while the console's generalist MLP is only modestly
better than guessing the average wear on unseen NASA runs. Trust it to **rank**
settings, not to promise an exact wear value — the fuzzy gate makes the final call.

---

## 6. Data

| Dataset | What it is | Status |
| --- | --- | --- |
| **NASA Milling** | 167 runs, 6 channels (current / vibration / AE), V_B labels | wired — trains the MLP, and the NASA CNN baselines |
| **PHM 2010** | high-speed milling, 3-flute ball-nose, stainless ≈ HRC 52, 7 ch, 50 kHz | wired — trains the PHM CNN-BiLSTM |
| **Vicomtech** | CNC turning, ~2,054 segments, control + sensor signals | wired — trains the Vicomtech HGBR |
| **LUH / KIT multivariate** | cast-iron shoulder milling, force + control deviation, VB 0–150 µm | wired — trains the LUH HGBR |
| **Synthetic** | a wide-envelope mechanics generator | used to blend the MLP's training set |
| MATWI, MU-TCM, QIT-CEMC | larger public datasets | catalogued in `docs/DATASET_INVENTORY.md`, **not** wired (future work) |

Raw and processed data live under `data/` and are git-ignored; the inventory and
provenance notes are in `docs/DATASET_INVENTORY.md` and `docs/VALIDATION.md`.

---

## 7. What is verified

Everything below is reproducible from the repo.

| Check | Result |
| --- | --- |
| `pytest` | **178 passed**, 0 failed, 0 skipped |
| `scripts/checks/check_web.py` (live endpoints) | **97** endpoint calls, 0 failures |
| `scripts/checks/validate_pipeline.py` | **1,876** invariants passed, 0 failures |
| `scripts/checks/stress_test.py 300` | **2,565** invariants passed, 0 failures, 0 anomalies |
| CLIPS matrix coverage | all **75** combinations of operation × ISO group × machine class match |
| Rule-engine agreement | the CLIPS engine and the readable rule spec agree on the whole matrix |

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\checks\validate_pipeline.py
.\.venv\Scripts\python.exe scripts\checks\stress_test.py 300
.\.venv\Scripts\python.exe scripts\checks\check_web.py   # needs the web console running
```

---

## 8. Honest limitations

- **No cross-lab validation.** Each model was validated only inside its own
  dataset's regime; zero-shot transfer of the steel MLP to another material
  fails. The lab models are shown, not silently relied on.
- The console's MLP is a **ranking aid**, not a wear gauge: on unseen runs it is
  only modestly better than predicting the mean.
- Some labels are **imputed, not measured** (e.g. NASA roughness is derived by
  formula); a few datasets have known signal warts.
- Empty Pareto fronts are a **legitimate answer** (some material/finish targets
  have no safe setting); the UI says so with the reason rather than inventing a
  point.
- The ISO 13399 catalogue is an **illustrative, format-accurate** set (50
  inserts), not a vendor export.

All of this is expanded, with numbers, in `docs/VALIDATION.md`.

---

## 9. Repository layout

```
me project/
|-- README.md                 # this file
|-- AGENTS.md                 # conventions for working in the repo
|-- pyproject.toml / requirements.txt / Makefile
|-- docs/                     # ARCHITECTURE, MODULE_BREAKDOWN, TOOLS,
|   |                         #   VALIDATION, DATASET_INVENTORY, PROJECT, ...
|   `-- Hybrid_AI_Manufacturing.pdf
|-- src/hybrid_ai/
|   |-- core/                 # shared schemas + dataset loaders (NASA, PHM, LUH, ...)
|   |-- stages/               # the four stage engines, one package each
|   |   |-- clips/            #   symbolic rules + CAPP / setup plan
|   |   |-- ann/              #   wear & roughness models
|   |   |-- fuzzy/            #   safety diagnosis + Tool Health Index
|   |   `-- nsga2/            #   Pareto optimisation + Taylor life model
|   |-- tools/                # typed run() tools + TOOL_SPEC (LLM/MCP-ready)
|   |-- web/                  # FastAPI console (api.py + static/)
|   `-- dashboard/            # Streamlit UI + the demo script
|-- tests/                    # mirrors src/ layout
|-- scripts/training/         # one-off training recipes
|-- scripts/checks/           # verification helpers (validate, stress, check_web, ...)
|-- models/                   # trained .pt / .joblib checkpoints
|-- data/                     # raw/ + processed/ (git-ignored)
|-- notebooks/                # colab_train.py
`-- archive/agentic-setup/    # archived LLM-agent scaffolding (not part of the live tree)
```

The **tools layer** (`src/hybrid_ai/tools/`) is the prominent interface: each
tool exposes a typed `run()` function and a `TOOL_SPEC` dict that is directly
consumable by LLMs and MCP.

---

## 10. Tech stack

- **Symbolic AI:** `clipspy` (a real CLIPS engine) — `stages/clips/`
- **ML / numerics:** `PyTorch`, `scikit-learn`, `numpy`, `scipy`, `pandas`
- **Fuzzy:** `scikit-fuzzy` (trapezoids + Mamdani Tool Health Index) — `stages/fuzzy/`
- **Optimisation:** `pymoo` (NSGA-II, SBX/PM, exact hypervolume) — `stages/nsga2/`
- **Storage:** `SQLite` + `SQLAlchemy` (ISO 13399-style tooling catalogue)
- **Front-ends:** `Streamlit` + `Plotly`, and `FastAPI` + a vanilla-JS single page

All dependencies are open-source; pinned versions are in `requirements.txt`.

---

## 11. Documentation map

| Read this | To learn |
| --- | --- |
| `docs/ARCHITECTURE.md` | the four stages and why each exists |
| `docs/MODULE_BREAKDOWN.md` | who owns what, and the per-member deliverables |
| `docs/TOOLS.md` | the tool interface (for LLM / MCP use) |
| `docs/VALIDATION.md` | every number, and its honest caveats |
| `docs/DATASET_INVENTORY.md` | the public datasets, what is wired, what is future |
| `docs/PROJECT.md` / `docs/PROJECT_REPORT.md` | the full narrative write-up |

---

## 12. Team & course

A five-member team; see `docs/MODULE_BREAKDOWN.md` for ownership. The work is
aligned with course **ME6324**, covering expert systems, neural networks, fuzzy
logic and genetic algorithms in one integrated CAPP + optimisation + diagnosis
workflow.
