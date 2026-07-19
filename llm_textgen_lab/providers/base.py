"""Provider protocol and shared helpers."""

from __future__ import annotations

from typing import Any, Protocol

from ..types import GenerationConfig, GenerationResult


class Provider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        ...


def as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}
