# Module Breakdown

Five team members, four pipeline stages, one shared core. Each owner
delivers a subpackage, a notebook, and a test set, then plugs their
output into the shared dashboard.

## Owner map

| # | Owner | Module | Package path | Notebook |
|---|---|---|---|---|
| 1 | M1 — Knowledge Engineer | CLIPS rule base + tool selection | `src/hybrid_ai/stages/clips/` | `notebooks/01_clips.ipynb` |
| 2 | M2 — ML Engineer | ANN wear/roughness predictor | `src/hybrid_ai/stages/ann/` | `notebooks/02_ann.ipynb` |
| 3 | M3 — Fuzzy / Diagnostics | Fuzzy state classifier | `src/hybrid_ai/stages/fuzzy/` | `notebooks/03_fuzzy.ipynb` |
| 4 | M4 — Optimisation | NSGA-II Pareto search | `src/hybrid_ai/stages/nsga2/` | `notebooks/04_nsga2.ipynb` |
| 5 | M5 — Platform / Data | Shared core, SQLite loader, Streamlit dashboard | `src/hybrid_ai/core/`, `src/hybrid_ai/dashboard/`, `src/hybrid_ai/data/` | `notebooks/00_setup.ipynb` |

## Per-module deliverables

Every module owner ships:

1. **Package** under `src/hybrid_ai/<module>/` with a `__init__.py`
   and at least one public function or class.
2. **Notebook** under `notebooks/` showing the module running
   end-to-end on the shared synthetic fixture.
3. **Tests** under `tests/<module>/` using `pytest`.
4. **Doc** update in `docs/` if the public interface changes.

## Interface contracts (so the modules can be wired up in parallel)

The shared `core` module (owned by M5) defines the data classes that
flow between stages. Other modules import these; they do not invent
their own.

```python
# src/hybrid_ai/core/schemas.py (planned)
PartSpec         # material, operation, machine, tolerances
ToolCandidate    # insert code, ISO group, speed/feed envelope, geometry
OperatingPoint   # V_c, f, a_p
WearPrediction   # V_B, R_a, model confidence
FuzzyDiagnosis   # state in {safe, caution, critical}, explanation
ParetoSolution   # OperatingPoint + (MRR, tool_life, R_a)
```

## Integration points

- M1 -> M2: `ToolCandidate` list becomes an input filter for the ANN.
- M2 -> M3: `WearPrediction` is one of Fuzzy's inputs.
- M2 -> M4: predicted V_B / R_a define the NSGA-II objective functions.
- M3 -> M1: a `caution` or `critical` diagnosis triggers the feedback
  loop back to CLIPS for re-planning.
- All -> M5: every stage exposes a `to_dict()` method that the
  Streamlit dashboard renders directly.

## Coordination cadence

- Weekly integration meeting: each owner demos the latest notebook
  against the shared fixture.
- M5 owns the dashboard and merges all branches.
- Any change to a shared schema in `core/` requires a heads-up in
  the team chat.