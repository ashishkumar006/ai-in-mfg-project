# Tools reference

The tools layer (`src/hybrid_ai/tools/`) is the prominent interface of
this project — for dashboards, the API, LLM agents and MCP clients
alike. Each tool is one module with a typed `run()` and a `TOOL_SPEC`
dict (name, description, JSON parameter schema). Tools call the stage
engines under `src/hybrid_ai/stages/`; they never duplicate logic.

See live specs any time: `hybrid_ai.agent.registry.tool_table()`.

| Tool | Calls into | Does |
|---|---|---|
| `select_tool` | stages/clips | CAPP plan, ranked inserts, envelope, notes; per-rule breakdown + closest inserts on no-match |
| `predict_wear` | stages/ann (MLP) | V_B + R_a at one operating point (synthetic or measured sensor) |
| `compare_models` | stages/ann (all 5) | One row per model; reference-window rows carry ground truth |
| `diagnose_state` | stages/fuzzy | safe/caution/critical + explanation + memberships; tolerance-relative bands |
| `optimize_pareto` | stages/nsga2 | Non-dominated Vc/f/ap front; `safe_only` gate (default on) |
| `list_tools` | core catalog | Inserts, filterable by ISO group / operation |
| `model_benchmarks` | all models | Held-out RMSE/MAE table |

Shared loading (model singletons, sensor synthesis, reference windows)
lives in `tools/_models.py`; the error contract (`code` + `message`)
in `tools/errors.py`; argument validation in `tools.validate_args`.

Agent scaffolding lives in `src/hybrid_ai/agent/`: `prompts/` (system,
guardrails, escalation), `registry.py` (tool table built from specs —
the single source of truth), `loop.py` (bounded ReAct with a scripted
fake backend for tests).

## Model backends (who drives the loop)

`agent.factory.make_client()` picks by environment (`.env` supported,
never committed):

| Backend | Env | Notes |
|---|---|---|
| `ollama` (default, offline) | `HYBRID_AI_MODEL=qwen2.5:1.5b` | Local SLM; smoke-tested 2/2 scenarios |
| `gemini` | `HYBRID_AI_BACKEND=gemini`, `HYBRID_AI_GEMINI_MODEL=gemini-3.5-flash-lite`, `GEMINI_API_KEY=...` | Faster; same tool specs via function declarations |

Copy `.env.example` to `.env` and paste the key. A future fine-tuned
SLM slots in as another backend in `factory.py` — loop, prompts and
tests stay untouched.
