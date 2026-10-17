from __future__ import annotations

import json

import httpx
import pytest

from hybrid_ai.agent import factory
from hybrid_ai.agent.gemini_client import GeminiClient, GeminiError

SPECS = [{"name": "select_tool", "description": "pick",
          "parameters": {"type": "object"}}]
MSGS = [{"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"}]


def _client(payload: dict, status: int = 200) -> GeminiClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return GeminiClient(model="test-model", api_key="KEY",
                        transport=httpx.MockTransport(handler))


def test_gemini_function_call_parsed() -> None:
    payload = {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "select_tool",
                          "args": {"part": {"part_id": "X"}}}}]}}]}
    out = _client(payload).complete(MSGS, SPECS)
    assert out == {"tool": "select_tool", "args": {"part": {"part_id": "X"}}}


def test_gemini_text_answer_parsed() -> None:
    payload = {"candidates": [{"content": {"parts": [{"text": "hello"}]}}]}
    out = _client(payload).complete(MSGS, SPECS)
    assert out == {"answer": "hello"}


def test_gemini_roles_mapped() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "ok"}]}}]})

    client = GeminiClient(model="m", api_key="k",
                          transport=httpx.MockTransport(handler))
    out = client.complete(MSGS + [{"role": "assistant", "content": "thinking"}],
                          SPECS)
    assert out == {"answer": "ok"}
    assert seen["system_instruction"]["parts"][0]["text"] == "sys"
    roles = [c["role"] for c in seen["contents"]]
    assert roles == ["user", "model"]
    assert seen["tools"][0]["function_declarations"][0]["name"] == "select_tool"


def test_gemini_missing_key_rejected() -> None:
    with pytest.raises(GeminiError):
        GeminiClient(model="m", api_key="")


def test_gemini_sanitizes_union_types() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [
            {"text": "ok"}]}}]})

    client = GeminiClient(model="m", api_key="k",
                          transport=httpx.MockTransport(handler))
    client.complete(MSGS, [{
        "name": "t", "description": "d",
        "parameters": {"type": "object",
                       "properties": {
                           "a": {"type": ["string", "null"]},
                           "b": {"type": "object",
                                 "properties": {"c": {"type": ["integer", "null"]}},
                                 "required": []}},
                       "required": []}}])
    props = seen["tools"][0]["function_declarations"][0]["parameters"]["properties"]
    assert props["a"]["type"] == "string"
    assert props["b"]["properties"]["c"]["type"] == "integer"


def test_gemini_404_names_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "not found"}})

    client = GeminiClient(model="nope", api_key="KEY",
                          transport=httpx.MockTransport(handler))
    with pytest.raises(GeminiError, match="nope"):
        client.complete(MSGS, SPECS)


def test_factory_defaults_to_ollama(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_AI_BACKEND", "ollama")
    client = factory.make_client()
    assert type(client).__name__ == "OllamaClient"


def test_factory_rejects_unknown_backend(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_AI_BACKEND", "gemni")
    with pytest.raises(ValueError, match="unknown HYBRID_AI_BACKEND"):
        factory.make_client()


def test_ollama_client_reads_late_env(monkeypatch) -> None:
    from hybrid_ai.agent.ollama_client import OllamaClient

    monkeypatch.setenv("HYBRID_AI_MODEL", "late-model:9b")
    assert OllamaClient().model == "late-model:9b"


def test_dotenv_supports_export_prefix(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text("export FOO_BAR_XYZ=42\n", encoding="utf-8")
    monkeypatch.delenv("FOO_BAR_XYZ", raising=False)
    factory.load_dotenv(env)
    assert __import__("os").environ["FOO_BAR_XYZ"] == "42"


def test_factory_gemini_backend(monkeypatch) -> None:
    monkeypatch.setenv("HYBRID_AI_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "KEY")
    client = factory.make_client()
    assert type(client).__name__ == "GeminiClient"


def test_dotenv_loader(tmp_path, monkeypatch) -> None:
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="ABC123"\n# comment\nEMPTY=\n',
                   encoding="utf-8")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    factory.load_dotenv(env)
    assert __import__("os").environ["GEMINI_API_KEY"] == "ABC123"


def _ollama_client(payload: dict, status: int = 200):
    import httpx

    from hybrid_ai.agent.ollama_client import OllamaClient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return OllamaClient(model="m", transport=httpx.MockTransport(handler))


def test_ollama_tool_call_parsed() -> None:
    payload = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "select_tool",
                      "arguments": '{"part": {"part_id": "X"}}'}}]}}]}
    client = _ollama_client(payload)
    out = client.complete([{"role": "user", "content": "hi"}], SPECS)
    assert out == {"tool": "select_tool", "args": {"part": {"part_id": "X"}}}


def test_ollama_answer_parsed() -> None:
    payload = {"choices": [{"message": {"content": "done"}}]}
    out = _ollama_client(payload).complete(
        [{"role": "user", "content": "hi"}], SPECS)
    assert out == {"answer": "done"}


def test_ollama_bad_arguments_json_tolerated() -> None:
    payload = {"choices": [{"message": {"tool_calls": [
        {"function": {"name": "select_tool", "arguments": "{oops"}}]}}]}
    out = _ollama_client(payload).complete(
        [{"role": "user", "content": "hi"}], SPECS)
    assert out == {"tool": "select_tool", "args": {}}
