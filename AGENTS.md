# AGENTS.md — Conventions for this project

This file tells assistants (and humans) how to work inside this repo
consistently.

## Language & runtime

- **Python 3.11+**.
- Dependencies pinned in `requirements.txt` (to be added in phase 1).
- All code targets Windows + PowerShell 5.1 (the development shell), but
  must remain cross-platform.

## Code style

- PEP 8, 4-space indentation, max line length 100.
- Type hints on all public function signatures.
- `from __future__ import annotations` at the top of every module.
- Docstrings: NumPy style for functions, Google style for classes.
- No comments unless they explain *why*, not *what*. Keep them sparse.

## Project layout rules

- All implementation code lives under `src/hybrid_ai/`.
- Each pipeline stage (CLIPS, ANN, Fuzzy, NSGA-II) gets its own
  subpackage: `clips/`, `ann/`, `fuzzy/`, `nsga2/`.
- Shared data structures live in `src/hybrid_ai/core/`.
- Streamlit dashboard lives in `src/hybrid_ai/dashboard/`.
- Tests mirror the package layout under `tests/`.

## Dependency policy

- Prefer mature, actively maintained libraries already named in the
  PDF (slide 5): `clipsPy`, `PyTorch`, `scikit-learn`, `scikit-fuzzy`,
  `DEAP`/`pymoo`, `SQLAlchemy`, `Streamlit`, `Plotly`.
- Do **not** add a dependency without recording it in `requirements.txt`.

## Run & test commands

These will become valid once the package is created in phase 1. Update
this section as scripts are added.

```powershell
# create venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# run tests
pytest

# run dashboard
streamlit run src/hybrid_ai/dashboard/app.py
```

## Git hygiene

- Do not commit large data files; `data/` is gitignored except for
  README placeholders.
- Do not commit secrets, `.venv/`, or `__pycache__/`.
- Commit messages: short imperative summary, optional body.

## What to read first when joining the project

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/MODULE_BREAKDOWN.md`
4. The doc under `docs/` matching the stage you own.