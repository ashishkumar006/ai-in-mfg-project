# Hybrid AI-Expert System for Automated Manufacturing

A modular pipeline that fuses a rule-based expert system, neural networks,
fuzzy logic, and a multi-objective evolutionary optimizer to support
autonomous CNC decision making: **CAPP**, **cutting-tool selection**, and
**machining optimization**.

## Why hybrid?

| Approach | Limitation |
| --- | --- |
| Static handbook lookup | Cannot react to real-time tool degradation |
| Pure ML model | Cannot infer operation sequences, justify tool choice, or explain chatter/chipping faults |
| Real machining | Flank wear (V_B) and surface roughness (R_a) evolve nonlinearly; MRR conflicts with tool life and finish |

The pipeline therefore combines symbolic reasoning, learned prediction,
uncertainty handling, and Pareto optimization in one loop.

## Pipeline at a glance

```
Part Specs -> CLIPS (symbolic) -> ANN (prediction) -> Fuzzy (state) -> NSGA-II (Pareto) -> Streamlit Dashboard
                                             ^                                                |
                                             |________________________________________________|
                                                 feedback loop for re-diagnosis / re-planning
```

Full architecture and rationale: see `docs/ARCHITECTURE.md`.

## Repository layout

```
me project/
|-- README.md
|-- AGENTS.md
|-- docs/
|   |-- ARCHITECTURE.md
|   |-- MODULE_BREAKDOWN.md
|   |-- DATA_PLAN.md
|   |-- IMPLEMENTATION_PLAN.md
|   |-- VALIDATION.md
|   |-- DATASET_INVENTORY.md
|   |-- PIPELINE_WALKTHROUGH.md
|   `-- PROJECT_WALKTHROUGH.md
|-- src/
|   `-- hybrid_ai/        # Python package (fully implemented)
|-- data/                 # raw + processed datasets (gitignored)
|-- notebooks/            # exploration notebooks + Colab script
|-- tests/                # pytest suite (34 tests, all passing)
`-- models/               # trained .pt checkpoints (gitignored)
```

## Tech stack

- **Logic:** `clipsPy`
- **ML / numerics:** `PyTorch`, `scikit-learn`, `numpy`, `scipy`, `pandas`
- **Fuzzy:** `scikit-fuzzy`
- **Optimization:** `pymoo` (NSGA-II)
- **Storage:** `SQLite` + `SQLAlchemy`
- **Dashboard:** `Streamlit`, `Plotly`

All dependencies are open-source. See `requirements.txt` for pinned versions.

## Team

Five-member team. See `docs/MODULE_BREAKDOWN.md` for ownership and
deliverables per member.

## Syllabus alignment

Aligned with course code **ME6324** — covers expert systems, neural
networks, fuzzy logic, and genetic algorithms in a single integrated
CAPP + optimization + diagnosis workflow.

## Status

Fully implemented and tested. Run the demo or dashboard:

```powershell
cd "C:\Users\AISHWARYA\Downloads\me project"
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo
.\.venv\Scripts\streamlit.exe run src\hybrid_ai\dashboard\app.py
```

## Dashboards

Two frontends share the same pipeline:

| Dashboard | URL | Notes |
|---|---|---|
| **New web UI** (FastAPI + single page, dark theme) | http://127.0.0.1:8502 | Clickable Pareto points, full-report export, Big PHM model row |
| Streamlit (classic) | http://127.0.0.1:8501 | Original tabbed UI |

```powershell
# new web UI (from src/ so the package imports)
cd src
..\.venv\Scripts\python.exe -m uvicorn hybrid_ai.web.api:app --port 8502
```

The demo runs three parts (steel turning, aluminium milling, superalloy
turning) through all four stages, then saves an interactive 3D Pareto
plot plus presenter notes to `demo_outputs/`:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo --smoke  # ~20 s check
# open demo_outputs\pareto.html and demo_outputs\SUMMARY.md
```