"""Google Gen AI SDK adapter for Gemini thinking levels/budgets."""

from __future__ import annotations

import os
import time

from ..types import GenerationConfig, GenerationResult
from .base import as_dict


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError("Install the API extra to use the Gemini adapter") from exc
        self.model = model
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        from google.genai import types

        kwargs = {"max_output_tokens": config.max_new_tokens}
        thinking: dict[str, object] = {}
        if config.reasoning_effort:
            thinking["thinking_level"] = config.reasoning_effort
        elif config.reasoning_budget is not None:
            thinking["thinking_budget"] = config.reasoning_budget
        if thinking:
            kwargs["thinking_config"] = types.ThinkingConfig(**thinking)
        for key in ("temperature", "top_p", "top_k"):
            value = getattr(config, key)
            if value is not None and (key != "temperature" or config.provider_options.get("allow_sampling")):
                kwargs[key] = value
        if config.stop_strings:
            kwargs["stop_sequences"] = list(config.stop_strings)
        started = time.perf_counter()
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(**kwargs),
        )
        usage = as_dict(getattr(response, "usage_metadata", None))
        return GenerationResult(
            text=getattr(response, "text", ""),
            provider=self.name,
            model=self.model,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"request": kwargs},
        )
