# Archived agentic setup

Archived on 2026-09-16: LLM orchestration removed from the live
codebase in favour of the deterministic pipeline
(`CLIPS -> ANN -> Fuzzy -> NSGA-II` via `src/hybrid_ai/tools/`).

## Why archived

- Core value is deterministic and tested; the LLM loop added
  latency, cost, secret handling and non-reproducibility without
  improving RMSE or Pareto quality.
- Agent chat is not needed for CAPP / diagnosis / optimisation;
  forms calling `/api/capp|diagnose|pareto|compare` directly cover it.

## What is here

- `src/hybrid_ai/agent/` — `loop.py`, `factory.py`,
  `ollama_client.py`, `gemini_client.py`, `registry.py`,
  `prompts/` — moved with `git mv`, history preserved.
- `tests/agent/` — `test_loop.py`, `test_clients.py` — moved.
- `scripts/checks/slm_smoke.py` — Ollama smoke test — moved.
- `docs/AGENTIC_ARCHITECTURE.md` — original blueprint — moved.
- `.env.example` — `GEMINI_API_KEY` template — moved.
- `web/` — reference copies (not `git mv`, deterministic parts stay
  live): `api.agent_endpoints.py` (`/api/chat|agent_run|
  agent_run_stream`), `test_api.agent_tests.py`,
  `app.agent_console.js`, `index.agent_console.html`,
  `tools_init.agent_registry.py`, `TOOLS.agent_section.md`.

## Restore if needed

```powershell
git mv archive/agentic-setup/src/hybrid_ai/agent src/hybrid_ai/agent
git mv archive/agentic-setup/tests/agent tests/agent
git mv archive/agentic-setup/scripts/checks/slm_smoke.py scripts/checks/slm_smoke.py
git mv archive/agentic-setup/docs/AGENTIC_ARCHITECTURE.md docs/AGENTIC_ARCHITECTURE.md
git mv archive/agentic-setup/.env.example .env.example
# then restore web endpoints from web/api.agent_endpoints.py
```
