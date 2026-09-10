"""SLM/API client for the agent loop, backed by the Gemini API.

Uses ``generateContent`` with function declarations built straight
from the tool specs, so the model sees the same tools as any other
client. Requires ``GEMINI_API_KEY`` (put it in ``.env``, never commit
it). Model id comes from ``HYBRID_AI_GEMINI_MODEL``.
"""

from __future__ import annotations

import os

import httpx

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


class GeminiError(RuntimeError):
    """The Gemini API refused or failed the request (message carries why)."""


def _sanitize_schema(node):
    """Adapt our TOOL_SPEC schemas to Gemini function declarations.

    Gemini rejects union type lists like ``["string", "null"]`` —
    nullability comes from omitting the key from ``required`` instead.
    """
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "type" and isinstance(value, list):
                non_null = [t for t in value if t != "null"]
                out[key] = non_null[0] if non_null else "string"
            else:
                out[key] = _sanitize_schema(value)
        return out
    if isinstance(node, list):
        return [_sanitize_schema(item) for item in node]
    return node


class GeminiClient:
    """ModelClient over the Gemini generateContent endpoint."""

    def __init__(self, model: str | None = None, api_key: str | None = None,
                 timeout_s: float = 120.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.model = model or os.environ.get("HYBRID_AI_GEMINI_MODEL",
                                             "gemini-3.5-flash-lite")
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not set (paste it in .env).")
        self.timeout_s = timeout_s
        self._transport = transport

    def complete(self, messages: list[dict], specs: list[dict]) -> dict:
        system_texts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = []
        for m in messages:
            if m.get("role") == "system":
                continue
            role = "model" if m.get("role") == "assistant" else "user"
            contents.append({"role": role,
                             "parts": [{"text": str(m.get("content", ""))}]})
        declarations = [{"name": s["name"], "description": s.get("description", ""),
                         "parameters": _sanitize_schema(
                             s.get("parameters", {"type": "object"}))}
                        for s in specs]
        payload = {
            "system_instruction": {"parts": [{"text": "\n\n".join(system_texts)}]},
            "contents": contents,
            "tools": [{"function_declarations": declarations}],
            "tool_config": {"function_calling_config": {"mode": "AUTO"}},
            "generationConfig": {"temperature": 0.0},
        }
        url = f"{API_BASE}/models/{self.model}:generateContent?key={self.api_key}"
        with httpx.Client(timeout=self.timeout_s,
                          transport=self._transport) as client:
            response = client.post(url, json=payload)
        if response.status_code == 404:
            raise GeminiError(f"model {self.model!r} not found. "
                              f"Check HYBRID_AI_GEMINI_MODEL. "
                              f"API said: {response.text[:300]}")
        if response.status_code >= 400:
            raise GeminiError(f"Gemini API error {response.status_code}: "
                              f"{response.text[:300]}")
        try:
            parts = response.json()["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, ValueError) as exc:
            raise GeminiError(f"unreadable Gemini response: {exc}")
        for part in parts:
            call = part.get("functionCall")
            if call:
                args = call.get("args") or {}
                if not isinstance(args, dict):
                    args = {}
                return {"tool": call.get("name"), "args": args}
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        return {"answer": text}
