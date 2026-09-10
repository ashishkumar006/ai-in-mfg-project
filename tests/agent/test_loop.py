from __future__ import annotations

from hybrid_ai.agent.loop import run_task
from hybrid_ai.agent.registry import load_prompt, tool_table


class ScriptedClient:
    """Fake SLM: replays scripted replies, records what it was asked."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.calls = 0

    def complete(self, messages: list[dict], specs: list[dict]) -> dict:
        self.calls += 1
        assert specs, "tools must be offered to the client"
        return self.replies.pop(0)


PART = {"part_id": "X", "material": "m", "iso_group": "P",
        "operation": "drilling"}


def test_prompts_and_registry_load() -> None:
    assert "procedure" in load_prompt("system").lower()
    assert "Provenance" in load_prompt("guardrails")
    assert "critical" in load_prompt("escalation").lower()
    names = {t["name"] for t in tool_table()}
    assert {"select_tool", "diagnose_state", "optimize_pareto"} <= names


def test_happy_path_uses_tool_then_answers() -> None:
    client = ScriptedClient([
        {"tool": "select_tool", "args": {"part": PART}},
        {"answer": "Use SDMX090308-P."},
    ])
    out = run_task("pick a drill for P steel", client)
    assert out["truncated"] is False
    assert out["answer"] == "Use SDMX090308-P."
    assert out["transcript"][0]["tool"] == "select_tool"
    assert "SDMX090308-P" in str(out["transcript"][0]["observation"])


def test_bad_args_retry_then_succeed() -> None:
    client = ScriptedClient([
        {"tool": "select_tool", "args": {"part": {"part_id": "X"}}},
        {"tool": "select_tool", "args": {"part": PART}},
        {"answer": "done."},
    ])
    out = run_task("pick a drill", client)
    assert out["truncated"] is False
    assert "error" in out["transcript"][0]["observation"]


def test_unknown_tool_is_reported_not_crash() -> None:
    client = ScriptedClient([
        {"tool": "nope", "args": {}},
        {"answer": "recovered."},
    ])
    out = run_task("do it", client)
    assert out["answer"] == "recovered."
    assert "unknown tool" in str(out["transcript"][0]["observation"])


def test_repeated_failure_escalates() -> None:
    client = ScriptedClient([
        {"tool": "optimize_pareto",
         "args": {"part": PART, "pop_size": 5}},
        {"tool": "optimize_pareto",
         "args": {"part": PART, "pop_size": 5}},
    ])
    out = run_task("optimise", client)
    assert out["truncated"] is True
    assert "Escalating" in out["answer"]
    assert client.calls == 2


def test_repeated_shape_failure_escalates() -> None:
    client = ScriptedClient([
        {"tool": "select_tool", "args": {"part": {"part_id": "X"}}},
        {"tool": "select_tool", "args": {"part": {"part_id": "X"}}},
    ])
    out = run_task("pick", client)
    assert out["truncated"] is True
    assert "Escalating" in out["answer"]


def test_max_steps_truncates() -> None:
    client = ScriptedClient([{"tool": "list_tools", "args": {}}] * 10)
    out = run_task("list", client, max_steps=3)
    assert out["truncated"] is True
    assert out["steps"] == 3
    assert "budget exhausted" in out["answer"]


def test_on_step_called_per_tool() -> None:
    seen: list = []
    client = ScriptedClient([
        {"tool": "list_tools", "args": {}},
        {"answer": "done."},
    ])
    out = run_task("list", client, on_step=seen.append)
    assert out["truncated"] is False
    assert len(seen) == 1 and seen[0]["tool"] == "list_tools"


class ExplodingClient:
    def complete(self, messages: list[dict], specs: list[dict]) -> dict:
        raise ConnectionError("backend down")


def test_backend_failure_escalates_gracefully() -> None:
    out = run_task("do it", ExplodingClient())
    assert out["truncated"] is True
    assert "Escalating" in out["answer"]
    assert out["transcript"][0]["backend_error"] == "backend down"


class SpinningClient:
    def complete(self, messages: list[dict], specs: list[dict]) -> dict:
        return {"tool": "list_tools", "args": {}}


def test_time_budget_escalates() -> None:
    out = run_task("do it", SpinningClient(), max_steps=1000, timeout_s=0.0)
    assert out["truncated"] is True
    assert "time budget" in out["answer"]


def test_missing_tools_nudges_before_answer() -> None:
    seen_messages: list = []

    class NudgedClient:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[dict], specs: list[dict]) -> dict:
            self.calls += 1
            seen_messages.extend(messages)
            if self.calls == 1:
                return {"answer": "early answer."}
            return {"tool": "list_tools", "args": {}}

    client = NudgedClient()
    out = run_task("list", client, max_steps=5, expect_tools=["list_tools"])
    assert out["truncated"] is True
    assert any("list_tools" in str(m) and "Not done yet" in str(m)
               for m in seen_messages)


def test_compliant_run_needs_no_nudge() -> None:
    client = ScriptedClient([
        {"tool": "list_tools", "args": {}},
        {"answer": "done."},
    ])
    out = run_task("list", client, expect_tools=["list_tools"])
    assert out["truncated"] is False
    assert out["steps"] == 1


def test_full_chain_select_diagnose_optimize() -> None:
    point = {"v_c_mpm": 250.0, "f_mmpr": 0.30, "ap_mm": 1.75}
    client = ScriptedClient([
        {"tool": "select_tool", "args": {"part": PART}},
        {"tool": "diagnose_state",
         "args": {"part": PART, "point": point}},
        {"tool": "optimize_pareto",
         "args": {"part": PART, "pop_size": 8, "n_generations": 3}},
        {"answer": "front ready."},
    ])
    out = run_task("full pipeline for the part", client, max_steps=8)
    assert out["truncated"] is False
    assert out["answer"] == "front ready."
    tools_used = [t["tool"] for t in out["transcript"] if "tool" in t]
    assert tools_used == ["select_tool", "diagnose_state", "optimize_pareto"]
    pareto_obs = str(out["transcript"][2]["observation"])
    assert "solutions" in pareto_obs
