"""Bounded tool-calling loop (ReAct skeleton, model-agnostic).

A client implements ``complete(messages, specs)`` and returns either
``{"tool": name, "args": {...}}`` or ``{"answer": text}``. The loop
validates, executes (max ``max_steps`` tool calls), retries bad args
once per tool, and returns a traceable transcript.
"""

from __future__ import annotations

import time
from typing import Protocol

from hybrid_ai.core.schemas import OperatingPoint, PartSpec
from hybrid_ai.tools import REGISTRY, validate_args
from hybrid_ai.tools.errors import ToolError


class ModelClient(Protocol):
    def complete(self, messages: list[dict], specs: list[dict]) -> dict: ...


def run_task(task: str, client: ModelClient, max_steps: int = 8,
             timeout_s: float | None = None, on_step=None,
             expect_tools: list[str] | None = None) -> dict:
    """Run one task through tools and return the transcript + answer.

    ``timeout_s`` bounds the whole run (not just one model call);
    ``on_step`` (optional) is called with each transcript entry as it
    is appended, enabling live streaming of the run. ``expect_tools``
    (optional) names tools that must succeed before an answer is
    accepted — an early answer triggers one nudge naming the missing
    tools instead of being returned.
    """
    from hybrid_ai.agent.registry import load_prompt, tool_table

    system = (load_prompt("system") + "\n\n"
              + load_prompt("guardrails") + "\n\n"
              + load_prompt("escalation"))
    messages: list[dict] = [{"role": "system", "content": system},
                            {"role": "user", "content": task}]
    specs = tool_table()
    transcript: list[dict] = []
    failures: dict[str, int] = {}
    steps = 0
    started = time.monotonic()
    while steps < max_steps:
        if timeout_s is not None and time.monotonic() - started > timeout_s:
            transcript.append({"timeout_s": timeout_s})
            return {"answer": "Stopped: time budget exhausted. "
                              "Escalating to the operator.",
                    "transcript": transcript, "steps": steps,
                    "truncated": True}
        try:
            call = client.complete(messages, specs)
        except Exception as exc:
            transcript.append({"backend_error": str(exc)})
            return {"answer": "Stopped: the model backend failed "
                              f"({exc}). Escalating to the operator.",
                    "transcript": transcript, "steps": steps,
                    "truncated": True}
        if "answer" in call:
            missing = _missing_tools(transcript, expect_tools)
            if missing:
                messages.append({"role": "assistant",
                                 "content": str(call["answer"])})
                messages.append({"role": "user",
                                 "content": ("Not done yet: these required tools "
                                             "have no successful result: "
                                             + ", ".join(missing)
                                             + ". Call them before answering.")})
                steps += 1
                continue
            transcript.append({"answer": call["answer"]})
            return {"answer": call["answer"], "transcript": transcript,
                    "steps": steps, "truncated": False}
        name, args = call.get("tool"), call.get("args", {})
        entry = REGISTRY.get(name or "")
        if entry is None:
            observation = {"error": f"unknown tool {name!r}",
                           "hint": "use one of: " + ", ".join(sorted(REGISTRY))}
        else:
            try:
                clean = validate_args(
                    entry["spec"], args if isinstance(args, dict) else {})
                result = entry["run"](**_bind(entry["spec"], clean))
                observation = {"result": _summarise(result)}
            except ToolError as exc:
                failures[name] = failures.get(name, 0) + 1
                if failures[name] >= 2:
                    return {"answer": f"Stopped: tool {name} failed twice "
                                      f"({exc.code}: {exc.message}). "
                                      f"Escalating to the operator.",
                            "transcript": transcript, "steps": steps,
                            "truncated": True}
                observation = {"error": exc.to_dict(),
                               "hint": "fix the arguments and retry once"}
            except TypeError as exc:
                failures[name] = failures.get(name, 0) + 1
                if failures[name] >= 2:
                    return {"answer": f"Stopped: tool {name} failed twice "
                                      f"(bad shape: {exc}). "
                                      f"Escalating to the operator.",
                            "transcript": transcript, "steps": steps,
                            "truncated": True}
                observation = {"error": str(exc),
                               "hint": "match the tool schema exactly"}
        transcript.append({"tool": name, "args": args, "observation": observation})
        if on_step is not None:
            on_step(transcript[-1])
        messages.append({"role": "assistant", "content": str({"tool": name, "args": args})})
        messages.append({"role": "user", "content": f"observation: {observation}"})
        steps += 1
    return {"answer": "Stopped: step budget exhausted. Escalating to the operator.",
            "transcript": transcript, "steps": steps, "truncated": True}


def _missing_tools(transcript: list[dict],
                   expect_tools: list[str] | None) -> list[str]:
    """Required tools with no successful (result-bearing) call yet."""
    if not expect_tools:
        return []
    succeeded = set()
    for entry in transcript:
        obs = entry.get("observation") or {}
        if entry.get("tool") and isinstance(obs, dict) and "result" in obs:
            succeeded.add(entry["tool"])
    return [name for name in expect_tools if name not in succeeded]


def _bind(spec: dict, args: dict) -> dict:
    """Coerce JSON-style arguments to the native objects tools take."""
    args = dict(args)
    if isinstance(args.get("part"), dict):
        args["part"] = PartSpec(**args["part"])
    if isinstance(args.get("point"), dict):
        args["point"] = OperatingPoint(**args["point"])
    return args


def _summarise(result, limit: int = 2000) -> object:
    text = str(result)
    return text if len(text) <= limit else text[:limit] + "…[truncated]"
