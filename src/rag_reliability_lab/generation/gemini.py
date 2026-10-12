"""Google Gemini generator, using the official ``google-genai`` SDK (``pip install .[gemini]``).

This replaces the legacy LangChain wrapper. Failure handling:

* a missing API key raises ``ConfigError`` before any question runs;
* HTTP 408, 429 and 5xx responses, and transport errors such as timeouts and connection
  resets, are retried with exponential backoff up to ``max_retries`` times;
* other 4xx responses (bad key, unknown or retired model) fail immediately;
* an empty response (for example a safety block) raises ``GenerationError``.

The runner records each ``GenerationError`` against its question and carries on with the
rest of the run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from ..errors import ConfigError, GenerationError
from ..models import ScoredChunk
from . import GeneratorOutput, build_prompt

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class GeminiGenerator:
    name = "gemini"

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        timeout_s: float = 60.0,
        max_retries: int = 3,
        backoff_s: float = 2.0,
        api_key_env: str = "GOOGLE_API_KEY",
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model.strip():
            raise ConfigError("gemini.model must be set")
        if max_retries < 0 or timeout_s <= 0 or max_output_tokens < 1:
            raise ConfigError("gemini: max_retries >= 0, timeout_s > 0, max_output_tokens >= 1")
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self._sleep = sleep
        try:
            from google.genai import types
        except ImportError as exc:
            raise ConfigError(
                "Gemini needs the optional extra: pip install 'rag-reliability-lab[gemini]'"
            ) from exc
        self._config = types.GenerateContentConfig(
            temperature=temperature, max_output_tokens=max_output_tokens
        )
        if client is None:
            api_key = os.environ.get(api_key_env, "").strip()
            if not api_key:
                raise ConfigError(f"environment variable {api_key_env} is not set (Gemini API key)")
            from google import genai

            client = genai.Client(
                api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000))
            )
        self._client = client

    def _classify(self, exc: Exception) -> GenerationError:
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            status = int(exc.code) if exc.code is not None else None
            retryable = status in RETRYABLE_STATUS
            return GenerationError(
                f"Gemini API error {status}: {exc.message or exc}",
                retryable=retryable,
                status=status,
            )
        try:
            import httpx

            transport_error: tuple[type[BaseException], ...] = (httpx.TransportError,)
        except ImportError:  # pragma: no cover - httpx ships with google-genai
            transport_error = ()
        if isinstance(exc, (TimeoutError, ConnectionError, *transport_error)):
            return GenerationError(
                f"Gemini transport error: {type(exc).__name__}: {exc}", retryable=True
            )
        return GenerationError(f"Gemini call failed: {type(exc).__name__}: {exc}", retryable=False)

    def generate(self, question: str, passages: Sequence[ScoredChunk]) -> GeneratorOutput:
        prompt = build_prompt(question, passages)
        attempt = 0
        while True:
            try:
                response = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=self._config
                )
                break
            except Exception as exc:
                error = self._classify(exc)
                if not error.retryable or attempt >= self.max_retries:
                    raise error from exc
                self._sleep(self.backoff_s * (2**attempt))
                attempt += 1
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            raise GenerationError("Gemini returned an empty response (possibly blocked)")
        return GeneratorOutput(text=text, model=self.model, metadata={"attempts": attempt + 1})

    def describe(self) -> dict[str, object]:
        return {
            "provider": "gemini",
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
        }
