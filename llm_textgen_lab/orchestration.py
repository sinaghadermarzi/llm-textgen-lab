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


def extract_answer(text: str, *, marker: str = "Answer:") -> str:
    """Pull the final short answer out of a full reasoning trajectory for voting."""

    _, found, tail = text.rpartition(marker)
    lines = [line.strip() for line in (tail if found else text).splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0] if found else lines[-1]


def _vote_counts(candidates: Sequence[str], extract: Callable[[str], str] | None) -> Counter[str]:
    if not candidates:
        raise ValueError("candidates cannot be empty")
    return Counter((extract(candidate) if extract else candidate).strip() for candidate in candidates)


def majority_vote(candidates: Sequence[str], *, extract: Callable[[str], str] | None = None) -> str:
    """A transparent exact-match vote useful for short-answer tasks.

    Pass ``extract`` to vote on extracted final answers instead of whole texts;
    ties resolve toward the earliest candidate.
    """

    return _vote_counts(candidates, extract).most_common(1)[0][0]


def agreement_rate(candidates: Sequence[str], *, extract: Callable[[str], str] | None = None) -> float:
    """Fraction of candidates agreeing with the winning vote; a cheap confidence signal."""

    counts = _vote_counts(candidates, extract)
    return counts.most_common(1)[0][1] / len(candidates)


def self_consistency(
    generator: Generator,
    prompt: str,
    config: GenerationConfig,
    count: int = 5,
    *,
    extract: Callable[[str], str] | None = None,
) -> ExperimentRecord:
    """Sample independent trajectories and majority-vote their (extracted) answers."""

    candidates = generate_candidates(generator, prompt, config, count)
    texts = [candidate.text for candidate in candidates]
    chosen = majority_vote(texts, extract=extract)
    return ExperimentRecord(
        strategy="self-consistency",
        result=GenerationResult(
            text=chosen,
            provider=candidates[0].provider,
            model=candidates[0].model,
            metadata={"agreement": agreement_rate(texts, extract=extract)},
        ),
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


def best_of_n(
    generator: Generator,
    prompt: str,
    config: GenerationConfig,
    *,
    score: Callable[[str], float],
    count: int = 5,
) -> ExperimentRecord:
    """Keep the candidate a programmatic scorer likes best; higher scores win.

    The scorer stands in for a test suite, checker, or reward model. It runs
    outside the generator, so ``calls`` counts only generator calls; contrast
    with ``verifier_select``, where the judge is itself a model call. Ties keep
    the earliest candidate.
    """

    candidates = generate_candidates(generator, prompt, config, count)
    best = max(candidates, key=lambda candidate: score(candidate.text))
    return ExperimentRecord(strategy="best-of-n", result=best, calls=count)


def refine_loop(
    generator: Generator,
    prompt: str,
    config: GenerationConfig,
    *,
    accept: Callable[[str], bool],
    max_rounds: int = 2,
    critic_instruction: str = "Review the candidate for errors. Return concrete corrections and missing checks.",
) -> ExperimentRecord:
    """Draft, then critique-and-revise until an external check accepts or rounds run out.

    ``accept`` models an execute-and-fix loop's checker: a test suite, compiler,
    or validator. A production loop would also feed the checker's error output
    into the critique prompt. ``calls`` is 1 + 2 * rounds used.
    """

    if max_rounds < 0:
        raise ValueError("max_rounds cannot be negative")
    current = generator(prompt, config)
    rounds_used = 0
    while rounds_used < max_rounds and not accept(current.text):
        critique_prompt = f"{critic_instruction}\n\nOriginal task:\n{prompt}\n\nCandidate:\n{current.text}"
        critique = generator(critique_prompt, config)
        revision_prompt = f"Revise the candidate using the critique. Return only the improved answer.\n\nCandidate:\n{current.text}\n\nCritique:\n{critique.text}"
        current = generator(revision_prompt, config)
        rounds_used += 1
    return ExperimentRecord(
        strategy="refine-loop",
        result=current,
        calls=1 + 2 * rounds_used,
        budget_label=f"rounds={rounds_used}",
    )


def escalate(
    generator: Generator,
    prompt: str,
    configs: Sequence[GenerationConfig],
    *,
    accept: Callable[[str], bool],
    labels: Sequence[str] | None = None,
) -> ExperimentRecord:
    """Try configurations cheapest-first and stop at the first accepted answer.

    Order ``configs`` by cost, for example ascending ``reasoning_effort`` or
    ``reasoning_budget``. When nothing is accepted the last attempt is returned;
    the caller holds ``accept`` and decides what that means. A cross-model
    cascade is a generator closure that dispatches on the config.
    """

    if not configs:
        raise ValueError("configs cannot be empty")
    tier_labels = list(labels) if labels is not None else [f"tier-{i}" for i in range(len(configs))]
    if len(tier_labels) != len(configs):
        raise ValueError("labels must match configs")
    for index, (config, label) in enumerate(zip(configs, tier_labels)):
        result = generator(prompt, config)
        if accept(result.text):
            return ExperimentRecord(strategy="escalate", result=result, calls=index + 1, budget_label=label)
    return ExperimentRecord(strategy="escalate", result=result, calls=len(configs), budget_label=tier_labels[-1])
