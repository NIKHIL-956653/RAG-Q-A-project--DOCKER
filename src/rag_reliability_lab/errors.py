"""Typed exceptions. Callers can distinguish bad input from runtime failures."""

from __future__ import annotations


class LabError(Exception):
    """Base class for all errors raised by the lab."""


class ConfigError(LabError):
    """A run configuration is missing, malformed or out of range."""


class DatasetError(LabError):
    """An evaluation dataset failed validation. ``problems`` lists every issue found."""

    def __init__(self, message: str, problems: list[str] | None = None) -> None:
        self.problems = list(problems or [])
        detail = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{message}\n{detail}" if detail else message)


class IngestError(LabError):
    """The corpus could not be read."""


class GenerationError(LabError):
    """A generator failed to produce an answer (API error, timeout, empty response)."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None) -> None:
        self.retryable = retryable
        self.status = status
        super().__init__(message)
