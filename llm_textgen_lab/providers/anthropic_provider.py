"""Anthropic Messages API adapter with adaptive/manual thinking modes."""

from __future__ import annotations

import os
import time

from ..types import GenerationConfig, GenerationResult
from .base import as_dict


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install the API extra to use the Anthropic adapter") from exc
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        request = {"model": self.model, "max_tokens": config.max_new_tokens, "messages": [{"role": "user", "content": prompt}]}
        if config.reasoning_budget is not None:
            request["thinking"] = {"type": "enabled", "budget_tokens": config.reasoning_budget}
        elif config.reasoning_effort:
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": config.reasoning_effort}
        if config.provider_options.get("thinking_display"):
            request.setdefault("thinking", {})["display"] = config.provider_options["thinking_display"]
        if config.temperature is not None and not request.get("thinking"):
            request["temperature"] = config.temperature
        if config.top_p is not None and config.provider_options.get("allow_sampling"):
            request["top_p"] = config.top_p
        if config.top_k is not None and config.provider_options.get("allow_sampling"):
            request["top_k"] = config.top_k
        if config.stop_strings:
            request["stop_sequences"] = list(config.stop_strings)
        started = time.perf_counter()
        response = self.client.messages.create(**request)
        blocks = getattr(response, "content", [])
        text = "".join(getattr(block, "text", "") for block in blocks if getattr(block, "type", "") == "text")
        thinking = "".join(getattr(block, "thinking", "") for block in blocks if getattr(block, "type", "") == "thinking")
        return GenerationResult(
            text=text,
            provider=self.name,
            model=self.model,
            usage=as_dict(getattr(response, "usage", None)),
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"request": request, "thinking_summary": thinking},
        )
