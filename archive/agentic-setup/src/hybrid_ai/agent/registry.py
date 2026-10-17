"""Single source of truth for what the agent may call.

Built from ``hybrid_ai.tools.REGISTRY`` so tool definitions cannot
drift between the pipeline, the API and the agent.
"""

from __future__ import annotations

from pathlib import Path

from hybrid_ai.tools import REGISTRY

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(name: str) -> str:
    """Read a prompt file (e.g. ``load_prompt("system")``)."""
    return (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def tool_table() -> list[dict]:
    """Name + description + parameters for every registered tool."""
    return [{"name": name, **entry["spec"]}
            for name, entry in sorted(REGISTRY.items())]
