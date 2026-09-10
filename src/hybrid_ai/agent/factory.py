"""Backend selection for the agent loop (no code changes to switch).

Environment::

    HYBRID_AI_BACKEND=ollama|gemini      (default: ollama)
    HYBRID_AI_MODEL=qwen2.5:1.5b         (Ollama model)
    HYBRID_AI_GEMINI_MODEL=gemini-3.5-flash-lite
    GEMINI_API_KEY=...                   (or paste it in .env)

``make_client()`` reads ``.env`` (repo root) first, so local runs need
no exports. Ollama stays the offline default; Gemini is one variable
away. A future fine-tuned SLM slots in as just another backend here.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE, # comments, single/double quotes).

    Only fills variables that are not already set. No dependency.
    """
    candidate = Path(path) if path else _find_dotenv()
    if candidate is None or not candidate.exists():
        return
    for line in candidate.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key.lower().startswith("export "):
            key = key[7:].strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _find_dotenv() -> Path | None:
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / ".env"
        if candidate.exists():
            return candidate
        if (parent / "pyproject.toml").exists():
            return parent / ".env"
    return None


def make_client():
    """Build the configured model client (ollama default, gemini opt-in)."""
    load_dotenv()
    backend = os.environ.get("HYBRID_AI_BACKEND", "ollama").strip().lower()
    if backend == "gemini":
        from hybrid_ai.agent.gemini_client import GeminiClient

        return GeminiClient()
    if backend == "ollama":
        from hybrid_ai.agent.ollama_client import OllamaClient

        return OllamaClient()
    raise ValueError(f"unknown HYBRID_AI_BACKEND {backend!r} "
                     f"(expected ollama|gemini)")
