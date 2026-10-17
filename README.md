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
|-- README.md / AGENTS.md / Makefile / pyproject.toml / requirements.txt
|-- docs/                 # ARCHITECTURE, MODULE_BREAKDOWN, TOOLS, VALIDATION, ...
|   `-- Hybrid_AI_Manufacturing.pdf
|-- src/hybrid_ai/        # core/ stages/ tools/ web/ dashboard/
|-- tests/                # mirrors src (151 tests, all passing on .venv)
|-- scripts/training|checks/
|-- archive/agentic-setup/  # archived LLM loop (see its README)
|-- vendor/unrar/         # UnRAR.exe + LICENSE (unreferenced, Windows-only)
|-- data/                 # raw/ + processed/ (gitignored) + README
|-- models/               # trained .pt/.joblib checkpoints
|-- notebooks/            # colab_train.py
`-- demo_outputs/         # regenerated demo artifacts (gitignored)
```

## Tech stack

- **Logic:** forward-chaining rules engine (`stages/clips/`, clipsPy-style API)
- **ML / numerics:** `PyTorch`, `scikit-learn`, `numpy`, `scipy`, `pandas`
- **Fuzzy:** trapezoidal membership + Mamdani rules (`stages/fuzzy/`)
- **Optimization:** `pymoo` (NSGA-II, SBX/PM, exact hypervolume indicator)
- **Storage:** `SQLite` + `SQLAlchemy`
- **Dashboard:** `Streamlit` + `Plotly`, and a FastAPI web UI

All dependencies are open-source. See `requirements.txt` for pinned
versions. (`clipsPy` and `scikit-fuzzy` are pinned there but not
currently imported — the rules and fuzzy stages are self-contained.)

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
| **New web UI** (FastAPI + single page, dark theme) | http://127.0.0.1:8502 | Clickable Pareto points, full-report export, Big PHM model row, deterministic pipeline runs |
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