"""Provider factory with no imports until a provider is requested."""

from __future__ import annotations

from typing import Any


def create_provider(name: str, model: str, **kwargs: Any):
    normalized = name.lower()
    if normalized == "openai":
        from .openai_provider import OpenAIProvider

        return OpenAIProvider(model, **kwargs)
    if normalized in {"anthropic", "claude"}:
        from .anthropic_provider import AnthropicProvider

        return AnthropicProvider(model, **kwargs)
    if normalized in {"gemini", "google"}:
        from .gemini_provider import GeminiProvider

        return GeminiProvider(model, **kwargs)
    raise ValueError("provider must be openai, anthropic, or gemini")
