# Agentic Architecture — Blueprint

*How this system evolves from a verified pipeline with dashboards into
an agent-orchestrated manufacturing assistant — by addition, not by
rewrite. Status: tools layer + agent scaffolding built; MCP server and
SLM smoke test remain (Phases 0–1 below).* 

---

## 0. The decision this document records

**We are NOT rewriting the pipeline, and we are NOT archiving the
codebase.** The CLIPS rules, trained models, fuzzy diagnoser, optimiser,
loaders, tests and dashboards stay exactly where they are, keep
passing, and keep serving. Reason, stated plainly:

1. The agent needs *tools*, and tools must be deterministic,
   typed, and measured. That is precisely what the current code is
   (151 tests, held-out-cutter metrics, Pydantic schemas on every
   boundary). Rewriting it would trade validated behavior for a
   blank page.
2. The industry's dominant agentic failure mode is the big-bang
   rebuild (Gartner: 40%+ cancelled by end of 2027). The winners
   add a thin orchestration layer over grounded tools.
3. This repo has no version control. "Archiving everything" without
   git means moving working code into a dead folder on one laptop —
   unrecoverable if anything goes wrong. If a snapshot is wanted,
   the safe version is `git init` + first commit, not deletion.

What we *are* building is a new top layer: an MCP tool surface over
the existing endpoints, plus an SLM orchestrator that drives them.
Everything below keeps working untouched, including both dashboards.

---

## 1. Target architecture

```
                        ┌─────────────────────────┐
  "200 alu brackets     │      AGENT (SLM)        │  narrates, sequences,
   by Friday" ───────▶  │  orchestrator + guard   │  retries, escalates
                        └───────────┬─────────────┘  (no physics of its own)
                    MCP tools (typed JSON, stateless)
        ┌───────────────┼───────────────────────┬───────────────┐
        ▼               ▼                       ▼               ▼
   select_tool    predict_wear /          diagnose        optimize_pareto
   get_capp       compare_models          (fuzzy)         (NSGA-II, gated)
        │               │                       │               │
        └───────────────┴───────────────────────┴───────────────┘
                        ▼
              CURRENT PIPELINE (frozen tool layer)
   stages/ (clips/ann/fuzzy/nsga2) + core/ — unchanged, still tested,
   still serving :8501 and :8502 exactly as today
```

Data flows one way for numbers: agent → tool → verified model →
validated JSON → agent narrates. No number in any agent reply may
originate anywhere else. That single rule is the whole safety case.

---

## 2. MCP tool surface (spec — thin wrap over existing endpoints)

Each tool maps 1:1 to a current FastAPI route, reusing its Pydantic
models unchanged. New package: `src/hybrid_ai/mcp_server/` (FastMCP).

| MCP tool | Source route | Input (existing schema) | Output |
|---|---|---|---|
| `select_tool` | `POST /api/capp` | `PartIn` | ranked inserts, envelope, plan, no-match diagnosis |
| `predict_wear` | `POST /api/compare` | `DiagnoseIn` | per-model V_B/R_a/uncertainty rows |
| `diagnose_state` | `POST /api/diagnose` | `DiagnoseIn` (+sensor_source) | state, explanation, memberships |
| `optimize_pareto` | `POST /api/pareto` | `ParetoIn` (safe_only default true) | solutions + extremes |
| `list_tools` | `GET /api/catalog` | filters | inserts |
| `benchmarks` | `GET /api/model_metrics` | — | held-out metrics table |

Read-only by construction: no tool writes, moves, or commands
anything. The agent physically cannot crash a machine through this
surface — a deliberate safety property, not a missing feature.

---

## 3. Agent loop design (`src/hybrid_ai/agent/`)

- **Loop:** bounded ReAct, max 8 tool steps. System prompt fixes the
  procedure: parse request → `select_tool` → `diagnose_state` at
  midpoint → if not safe, `optimize_pareto` (safe-only) → confirm the
  chosen point → narrate with every number cited to a tool response.
- **Model:** local SLM via Ollama (shortlist: qwen3:1.7b, qwen2.5:3b —
  both score 0.8–0.96 on tool-calling benchmarks with restraint at
  1.0). Temperature 0, JSON-mode tool calls, retry-on-schema-failure.
  Model-agnostic client so an API model can substitute without code
  changes.
- **Escalation policy (mirrors the fuzzy gate):** safe → proceed;
  caution → proceed *with* a flagged warning; critical or empty front
  → stop, explain, propose the re-plan options, require human pick.
  This is the existing feedback loop with a narrator attached.

---

## 4. Guardrails (non-negotiable)

1. **Provenance rule:** every numeric claim traces to a tool response
   ID. Regex/JSON check before replying; violation → re-run, never
   patch numbers by hand.
2. **Schema validation** on every tool call AND every tool response
   (reuse the Pydantic models — no new validation code to drift).
3. **Bounded autonomy:** max steps, no writes, no shell, no network
   beyond localhost:8502.
4. **Uncertainty surfaced, not hidden:** ± spreads and caution states
   must appear in the final answer, never be smoothed over.
5. **Offline-first:** the pipeline, dashboards and tests work with
   zero model weights installed. The agent degrades to "unavailable",
   never to "guessing".

---

## 5. Evaluation harness (how we prove the agent, not just claim it)

Scripted scenarios with verifiable outcomes, e.g.: "P steel turning,
medium, Ra 1.2 → must return CNMG120408-PM with its envelope";
"exotic spec → must surface the no-match panel, never invent an
insert"; "critical midpoint → must reroute to safe-only Pareto and say
so". Metrics: task-success rate, number hallucination rate (target:
zero, enforced), sensor-source correctness, paraphrase robustness
(same scenario reworded 3 ways). The ground truth already exists in
`tests/` and `scripts/checks/check_web.py` — the harness reuses it.

---

## 6. Phased build plan

- **Phase 0 — MCP server** (`mcp` package, ~200 lines, tests with
  FastMCP client). Usable immediately from any MCP client. No model
  needed. *Ends with: tools callable from Claude Desktop/VS Code.*
- **Phase 1 — SLM smoke test.** Ollama + shortlisted SLM against
  :8502 on 5–10 scripted scenarios. Produces a go/no-go score.
  *Ends with: measured verdict, not a debate.*
- **Phase 2 — Agent package** with §3 loop + §4 guardrails + §5
  harness in CI (scripted fake SLM backend so tests need no weights).
  *Ends with: demo conversation, end to end.*
- **Phase 3 (later) — RAG over `docs/`** for knowledge questions;
  planner/diagnostician split only if one agent frays.

Explicitly deferred: fine-tuning an SLM (prompt first), UI rebuilds,
new datasets beyond NUAA, any change to the tool layer's semantics.

---

## 7. What stays frozen

   `stages/` (clips/ann/fuzzy/nsga2), `core/` (logic + schemas),
`web/api.py` routes and response shapes, both dashboards, all 82
tests, all checkpoints, `scripts/` recipes. Changes here need the
same bar as today: tests green, metrics reproduced, docs updated.
