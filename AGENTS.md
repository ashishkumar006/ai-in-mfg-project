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
- The **tools layer** (`src/hybrid_ai/tools/`, one module per tool)
  is the prominent interface: typed `run()` functions + `TOOL_SPEC`
  dicts consumable by LLMs and MCP. Reference tools first.
- Pipeline stage engines live **deep** under
  `src/hybrid_ai/stages/` (`clips/`, `ann/`, `fuzzy/`, `nsga2/`);
  touch them only when the logic itself must change.
- Shared data structures live in `src/hybrid_ai/core/`.
- Agent scaffolding (prompts, registry, loop) lives in
  `src/hybrid_ai/agent/`.
- Streamlit dashboard lives in `src/hybrid_ai/dashboard/`; the web
  API + frontend in `src/hybrid_ai/web/`.
- Tests mirror the package layout under `tests/`.
- One-off recipes live in `scripts/training/`; verification helpers
  in `scripts/checks/`. Both locate the repo root via `pyproject.toml`,
  never by relative depth.

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