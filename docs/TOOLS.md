# Tools reference

The tools layer (`src/hybrid_ai/tools/`) is the prominent interface of
this project — for dashboards, the API and scripts alike. Each tool
is one module with a typed `run()` and a `TOOL_SPEC` dict (name,
description, JSON parameter schema). Tools call the stage engines
under `src/hybrid_ai/stages/`; they never duplicate logic.

See live specs in `src/hybrid_ai/tools/__init__.py:REGISTRY`.

| Tool | Calls into | Does |
|---|---|---|
| `select_tool` | stages/clips | CAPP plan, ranked inserts, envelope, notes; per-rule breakdown + closest inserts on no-match |
| `predict_wear` | stages/ann (MLP) | V_B + R_a at one operating point (synthetic or measured sensor) |
| `compare_models` | stages/ann (all 5) | One row per model, **tiered** `production` (MLP, CNN-BiLSTM, HGBR) vs `reference` (CNN1D, CNN-LSTM — overfit, not used for decisions); `production_spread_mm` is the agreement figure |
| `diagnose_state` | stages/fuzzy | safe/caution/critical + explanation + memberships; tolerance-relative bands |
| `optimize_pareto` | stages/nsga2 | Non-dominated Vc/f/ap front; `safe_only` gate (default on). An empty front returns `no_safe_front` with the vetoing diagnosis and a hint — never a bare empty list. Also returns `hypervolume` (exact pymoo HV of the front), `spread` (Schott spacing CV + axis extent of the front) and `taylor_check` (ANN life vs the extended Taylor equation at each extreme) |
| `list_tools` | core catalog | Inserts, filterable by ISO group / operation |
| `model_benchmarks` | all models | Held-out RMSE/MAE table |
| `sweep_pareto` | stages/nsga2 (via `optimize_pareto`) | Read-only seed sweep; per-seed extremes + spread + stable/check/unstable verdict |
| `sensitivity_probe` | stages/fuzzy (via `diagnose_state`) | Read-only ±rel_delta probe per axis; state flips + most sensitive axis |
| `propose_recalibration` | stages/fuzzy (pure grid) | PROPOSAL-ONLY dry run: candidate thresholds vs current over a 100-cell grid; always `applied: false` |
| `replan` | stages/clips + ann + fuzzy | **Closed loop.** Given the insert (and point) that failed, re-runs the CLIPS stage (`build_capp_sheet`, clipspy engine, failing insert excluded), grid-searches the new envelope for the safest setting, and diagnoses it. Returns `stage: "clips"` + `engine` so callers can see the loop really goes through the rule stage. Bounded to one re-plan; reports `resolved`, a `reading` (`safe`/`improved`/`no_improvement`/`no_alternative`) and never invents a fix |

Tier-1 physics/model cross-checks:

- `stages/nsga2/taylor.py` — extended Taylor `V_c·T^n·D^z·f^y = K`
  tool-life model (per-ISO `n`,`K`), used only as an independent check
  on the ANN's `life = 6.0/V_B` proxy. `compare_life` returns
  `consistent` / `ann_optimistic` / `ann_pessimistic`.
- `scripts/training/heldout_nasa.py` — leave-one-run-out evaluation of
  the blended MLP on the 23 NASA runs (honest generalisation, not a
  random blended holdout). Writes `docs/heldout_nasa.json`.
- `scripts/checks/bench_speed.py` — per-stage latency + search-precision
  (NSGA-II gate on vs random sampling on equal budget).
- `core/tooling_catalog.py` — the deck's "catalog → parsed → SQLite
  (ISO-keyed)" step: parses `src/hybrid_ai/assets/tool_catalog_iso13399.csv`
  (50 inserts, 2 per (group, operation), validated per row), then
  `tools/_models.get_catalog()` serves it from the SQLite store at
  `data/processed/tool_catalog.sqlite`.

Shared loading (model singletons, sensor synthesis, reference windows)
lives in `tools/_models.py`; the error contract (`code` + `message`)
in `tools/errors.py`; argument validation in `tools.validate_args`.

Former LLM agent scaffolding is archived to
`archive/agentic-setup/` (loop, clients, prompts, registry, smoke
test). The live path needs no model backend and no secrets: call
`tools.*.run()` directly or via `POST /api/capp|diagnose|pareto|
compare|sweep|sensitivity|propose_recalibration|replan`.
