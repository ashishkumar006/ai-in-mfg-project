"""Agent scaffolding: prompts, tool registry view, bounded loop.

The loop is model-agnostic: any client implementing ``complete()``
works, including the scripted fake used in tests (no weights needed).
"""

from __future__ import annotations

from hybrid_ai.agent.factory import load_dotenv, make_client
from hybrid_ai.agent.loop import run_task
from hybrid_ai.agent.registry import tool_table

__all__ = ["load_dotenv", "make_client", "run_task", "tool_table"]
