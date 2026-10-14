"""Respondent adapter interface.

Every model (and the PRNG control) is a respondent exposing
``query(prompt, temperature, seed) -> text`` plus a ``fingerprint()`` that pins
exactly what answered. The harness never talks to a provider directly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class Respondent(ABC):
    def __init__(self, spec: dict, max_tokens: int = 64):
        self.spec = spec
        self.id = spec["id"]
        self.max_tokens = max_tokens

    @abstractmethod
    def query(self, prompt: str, temperature: float, seed: int | None = None) -> str:
        """Return the raw text of one response."""

    @abstractmethod
    def fingerprint(self) -> dict:
        """Return everything needed to say exactly which model answered."""

    @property
    def is_control(self) -> bool:
        return bool(self.spec.get("control"))


def make_respondent(spec: dict, max_tokens: int = 64) -> Respondent:
    provider = spec["provider"]
    if provider == "ollama":
        from respondents.ollama import OllamaRespondent
        return OllamaRespondent(spec, max_tokens)
    if provider == "prng":
        from respondents.prng import PRNGRespondent
        return PRNGRespondent(spec, max_tokens)
    if provider == "openai":
        from respondents.openai import OpenAIRespondent
        return OpenAIRespondent(spec, max_tokens)
    if provider == "anthropic":
        from respondents.anthropic import AnthropicRespondent
        return AnthropicRespondent(spec, max_tokens)
    if provider == "google":
        from respondents.google import GoogleRespondent
        return GoogleRespondent(spec, max_tokens)
    raise ValueError(f"unknown provider {provider!r}")
