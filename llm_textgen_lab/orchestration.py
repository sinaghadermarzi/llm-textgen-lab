"""Inference-time compute strategies built around any callable generator."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .types import GenerationConfig, GenerationResult

Generator = Callable[[str, GenerationConfig], GenerationResult]


@dataclass
class ExperimentRecord:
    strategy: str
    result: GenerationResult
    calls: int
    budget_label: str | None = None


def generate_candidates(generator: Generator, prompt: str, config: GenerationConfig, count: int) -> list[GenerationResult]:
    """Run independent candidates; use this to study self-consistency."""

    if count <= 0:
        raise ValueError("count must be positive")
    return [generator(prompt, config) for _ in range(count)]


def majority_vote(candidates: Sequence[str]) -> str:
    """A transparent exact-match vote useful for short-answer tasks."""

    if not candidates:
        raise ValueError("candidates cannot be empty")
    return Counter(candidate.strip() for candidate in candidates).most_common(1)[0][0]


def self_consistency(generator: Generator, prompt: str, config: GenerationConfig, count: int = 5) -> ExperimentRecord:
    candidates = generate_candidates(generator, prompt, config, count)
    chosen = majority_vote([candidate.text for candidate in candidates])
    return ExperimentRecord(
        strategy="self-consistency",
        result=GenerationResult(text=chosen, provider=candidates[0].provider, model=candidates[0].model),
        calls=count,
    )


def critic_revision(
    generator: Generator,
    prompt: str,
    config: GenerationConfig,
    *,
    critic_instruction: str = "Review the candidate for errors. Return concrete corrections and missing checks.",
) -> ExperimentRecord:
    draft = generator(prompt, config)
    critique_prompt = f"{critic_instruction}\n\nOriginal task:\n{prompt}\n\nCandidate:\n{draft.text}"
    critique = generator(critique_prompt, config)
    revision_prompt = f"Revise the candidate using the critique. Return only the improved answer.\n\nCandidate:\n{draft.text}\n\nCritique:\n{critique.text}"
    revised = generator(revision_prompt, config)
    return ExperimentRecord(strategy="draft-critic-revision", result=revised, calls=3)


def verifier_select(
    generator: Generator,
    prompt: str,
    config: GenerationConfig,
    candidates: Iterable[str],
    *,
    verifier_instruction: str = "Score each candidate for correctness. Return the index of the best candidate and one short reason.",
) -> ExperimentRecord:
    choices = list(candidates)
    if not choices:
        raise ValueError("candidates cannot be empty")
    joined = "\n\n".join(f"Candidate {i}:\n{text}" for i, text in enumerate(choices))
    verification = generator(f"{verifier_instruction}\n\nTask:\n{prompt}\n\n{joined}", config)
    return ExperimentRecord(strategy="verifier-select", result=verification, calls=1)
