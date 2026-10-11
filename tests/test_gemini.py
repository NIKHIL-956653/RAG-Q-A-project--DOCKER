"""Gemini provider tests with a fake client: no network, no API key, no cost."""

from __future__ import annotations

from typing import Any

import pytest

from rag_reliability_lab.errors import ConfigError, GenerationError

from .test_generation import PASSAGES

genai_errors = pytest.importorskip("google.genai.errors")


class FakeResponse:
    def __init__(self, text: str | None) -> None:
        self.text = text


class FakeModels:
    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return FakeResponse(item)


class FakeClient:
    def __init__(self, script: list[Any]) -> None:
        self.models = FakeModels(script)


def _api_error(code: int) -> Exception:
    return genai_errors.APIError(code, {"error": {"message": f"status {code}", "status": "X"}})


def _gemini(script: list[Any], **kwargs: Any) -> tuple[Any, FakeClient, list[float]]:
    from rag_reliability_lab.generation.gemini import GeminiGenerator

    client = FakeClient(script)
    sleeps: list[float] = []
    gen = GeminiGenerator("gemini-test", client=client, sleep=sleeps.append, **kwargs)
    return gen, client, sleeps


def test_gemini_success_sends_grounded_prompt() -> None:
    gen, client, sleeps = _gemini(["It buffers 72 hours [P1]."])
    out = gen.generate("How long?", PASSAGES[:1])
    assert out.text == "It buffers 72 hours [P1]."
    assert out.model == "gemini-test"
    call = client.models.calls[0]
    assert call["model"] == "gemini-test" and "[P1] source: doc.md" in call["contents"]
    assert call["config"].temperature == 0.0
    assert sleeps == []


def test_gemini_retries_rate_limits_and_server_errors() -> None:
    gen, client, sleeps = _gemini([_api_error(429), _api_error(503), "ok [P1]"], backoff_s=1.0)
    assert gen.generate("q", PASSAGES).text == "ok [P1]"
    assert sleeps == [1.0, 2.0]
    assert len(client.models.calls) == 3


def test_gemini_retries_transport_errors() -> None:
    httpx = pytest.importorskip("httpx")
    gen, _, sleeps = _gemini([httpx.ConnectTimeout("slow"), "ok"])
    assert gen.generate("q", PASSAGES).text == "ok"
    assert len(sleeps) == 1


def test_gemini_does_not_retry_client_errors() -> None:
    gen, client, sleeps = _gemini([_api_error(404), "never reached"])
    with pytest.raises(GenerationError) as info:
        gen.generate("q", PASSAGES)
    assert info.value.status == 404 and not info.value.retryable
    assert sleeps == [] and len(client.models.calls) == 1


def test_gemini_gives_up_after_max_retries() -> None:
    gen, client, sleeps = _gemini([_api_error(500)] * 3, max_retries=2)
    with pytest.raises(GenerationError, match="500"):
        gen.generate("q", PASSAGES)
    assert len(client.models.calls) == 3 and len(sleeps) == 2


def test_gemini_empty_response_is_an_error() -> None:
    gen, _, _ = _gemini([None])
    with pytest.raises(GenerationError, match="empty"):
        gen.generate("q", PASSAGES)


def test_gemini_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from rag_reliability_lab.generation.gemini import GeminiGenerator

    monkeypatch.delenv("RAG_LAB_TEST_KEY", raising=False)
    with pytest.raises(ConfigError, match="RAG_LAB_TEST_KEY"):
        GeminiGenerator("gemini-test", api_key_env="RAG_LAB_TEST_KEY")
    with pytest.raises(ConfigError):
        GeminiGenerator(" ", client=FakeClient([]))
