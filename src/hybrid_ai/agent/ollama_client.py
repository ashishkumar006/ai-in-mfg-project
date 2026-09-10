"""SLM client for the agent loop, backed by a local Ollama server.

Speaks the OpenAI-compatible chat API (``/v1/chat/completions``),
so any model Ollama serves works — local weights, no API keys, no
data leaving the machine. Set ``HYBRID_AI_MODEL`` to pick the model
(defaults to ``qwen2.5:1.5b``).
"""

from __future__ import annotations

import json
import os

import httpx


class OllamaClient:
    """ModelClient over Ollama's OpenAI-compatible endpoint."""

    def __init__(self, model: str | None = None,
                 base_url: str | None = None,
                 timeout_s: float = 180.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        # Resolved here (not at import) so .env / os.environ changes
        # made after import still take effect.
        self.model = model or os.environ.get("HYBRID_AI_MODEL", "qwen2.5:1.5b")
        self.base_url = (base_url or os.environ.get(
            "HYBRID_AI_BASE_URL", "http://127.0.0.1:11434/v1")).rstrip("/")
        self.timeout_s = timeout_s
        self._transport = transport

    def complete(self, messages: list[dict], specs: list[dict]) -> dict:
        tools = [{"type": "function",
                  "function": {"name": s["name"],
                               "description": s.get("description", ""),
                               "parameters": s.get("parameters",
                                                   {"type": "object"})}}
                 for s in specs]
        payload = {"model": self.model, "messages": messages,
                   "tools": tools, "tool_choice": "auto",
                   "temperature": 0.0}
        with httpx.Client(timeout=self.timeout_s,
                           transport=self._transport) as client:
            response = client.post(f"{self.base_url}/chat/completions",
                                   json=payload)
            response.raise_for_status()
            message = response.json()["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if calls:
            fn = calls[0]["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            return {"tool": fn.get("name"), "args": args}
        return {"answer": message.get("content") or ""}
