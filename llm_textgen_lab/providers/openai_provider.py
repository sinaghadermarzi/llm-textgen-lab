"""OpenAI Responses API adapter.

The import is deliberately lazy so the offline track works without the SDK.
"""

from __future__ import annotations

import os
import time

from ..types import GenerationConfig, GenerationResult
from .base import as_dict


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the API extra to use the OpenAI adapter") from exc
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        request = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": config.max_new_tokens,
        }
        if config.reasoning_effort:
            request["reasoning"] = {"effort": config.reasoning_effort}
            if config.provider_options.get("reasoning_summary"):
                request["reasoning"]["summary"] = config.provider_options["reasoning_summary"]
        if config.stop_strings and config.provider_options.get("allow_stop"):
            request["stop"] = list(config.stop_strings)
        # Reasoning models may reject sampling controls. Opt in explicitly for
        # a model/API combination known to support them.
        if config.provider_options.get("allow_sampling"):
            for key in ("temperature", "top_p"):
                value = getattr(config, key)
                if value is not None:
                    request[key] = value
        started = time.perf_counter()
        response = self.client.responses.create(**request)
        usage = as_dict(getattr(response, "usage", None))
        return GenerationResult(
            text=getattr(response, "output_text", ""),
            provider=self.name,
            model=self.model,
            usage=usage,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"request": request},
        )
