"""Small provider- and backend-neutral data structures."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationConfig:
    """Controls shared by hosted and local experiments.

    Not every provider supports every field. Adapters should map supported
    fields explicitly and preserve the rest in ``provider_options`` for
    experiments that need provider-specific behavior.
    """

    max_new_tokens: int = 256
    temperature: float | None = 0.7
    top_k: int | None = None
    top_p: float | None = 1.0
    min_p: float | None = None
    typical_p: float | None = None
    do_sample: bool = True
    num_beams: int = 1
    num_return_sequences: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    stop_strings: tuple[str, ...] = ()
    seed: int | None = 0
    reasoning_effort: str | None = None
    reasoning_budget: int | None = None
    provider_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive")
        for name, value in (("top_p", self.top_p), ("min_p", self.min_p), ("typical_p", self.typical_p)):
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.num_beams <= 0 or self.num_return_sequences <= 0:
            raise ValueError("beam and return counts must be positive")
        if self.num_return_sequences > self.num_beams and self.num_beams > 1:
            raise ValueError("num_return_sequences cannot exceed num_beams")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative")
        if self.reasoning_budget is not None and self.reasoning_budget <= 0:
            raise ValueError("reasoning_budget must be positive")

    def to_transformers_kwargs(self) -> dict[str, Any]:
        """Return only the common generation kwargs understood by HF."""

        result: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "num_return_sequences": self.num_return_sequences,
            "repetition_penalty": self.repetition_penalty,
        }
        optional = {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "typical_p": self.typical_p,
            "no_repeat_ngram_size": self.no_repeat_ngram_size or None,
            "stop_strings": self.stop_strings or None,
        }
        result.update({key: value for key, value in optional.items() if value is not None})
        return result


@dataclass
class GenerationResult:
    """Normalized result with raw provider metadata retained for study."""

    text: str
    provider: str = "unknown"
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
