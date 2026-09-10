"""SLM smoke test: real local model drives the agent loop (Ollama needed).

Scenarios with verifiable outcomes; prints pass/fail per scenario.

Usage (Ollama serving, e.g. qwen2.5:1.5b already pulled)::
    $env:PYTHONPATH = "src"
    $env:HYBRID_AI_MODEL = "qwen2.5:1.5b"
    .\\.venv\\Scripts\\python.exe scripts/checks/slm_smoke.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from hybrid_ai.agent.factory import make_client
from hybrid_ai.agent.loop import run_task

PART = {"part_id": "BRACKET-A1", "material": "AISI 1045 steel",
        "iso_group": "P", "operation": "drilling", "machine": "medium",
        "hardness_hb": 220.0, "target_ra_um": 1.6, "batch_size": 50}

SCENARIOS = [
    ("Select a tool for this part: " + str(PART),
     lambda out: any(t.get("tool") == "select_tool"
                     and "SDMX090308-P" in str(t.get("observation"))
                     for t in out["transcript"] if "tool" in t)),
    ("Diagnose the machining state for part " + str(PART) +
     " at Vc 170, feed 0.16, depth 1.75",
     lambda out: any(t.get("tool") == "diagnose_state"
                     and any(s in str(t.get("observation"))
                             for s in ("safe", "caution", "critical"))
                     for t in out["transcript"] if "tool" in t)),
]


def main() -> None:
    client = make_client()
    backend = type(client).__name__
    model = getattr(client, "model", "?")
    print(f"backend: {backend} | model: {model}")
    passed = 0
    for i, (task, check) in enumerate(SCENARIOS):
        t0 = time.time()
        try:
            out = run_task(task, client, max_steps=6)
            ok = check(out)
        except Exception as exc:
            ok, out = False, {"error": str(exc)}
        dt = time.time() - t0
        passed += bool(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] scenario {i + 1} ({dt:.0f}s): {task}")
        print(f"  answer: {str(out.get('answer'))[:220]}")
    print(f"{passed}/{len(SCENARIOS)} scenarios passed")


if __name__ == "__main__":
    main()
