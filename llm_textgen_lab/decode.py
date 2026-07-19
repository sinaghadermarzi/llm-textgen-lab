"""Readable token-level decoding helpers for toy experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass
class Beam:
    tokens: tuple[int, ...]
    score: float


def greedy_decode(next_logits: Callable[[tuple[int, ...]], Iterable[float]], *, steps: int, bos_token: int = 0) -> list[int]:
    """Select the highest-logit token at each step."""

    sequence = [bos_token]
    for _ in range(steps):
        logits = list(next_logits(tuple(sequence)))
        if not logits:
            break
        sequence.append(max(range(len(logits)), key=logits.__getitem__))
    return sequence


def beam_search(
    next_logits: Callable[[tuple[int, ...]], Iterable[float]],
    *,
    steps: int,
    width: int = 3,
    bos_token: int = 0,
) -> list[int]:
    """A tiny beam-search implementation for visualizing breadth vs depth."""

    if width <= 0:
        raise ValueError("width must be positive")
    beams = [Beam((bos_token,), 0.0)]
    for _ in range(steps):
        expanded: list[Beam] = []
        for beam in beams:
            for token, logit in enumerate(next_logits(beam.tokens)):
                expanded.append(Beam(beam.tokens + (token,), beam.score + float(logit)))
        beams = sorted(expanded, key=lambda item: item.score, reverse=True)[:width]
    return list(max(beams, key=lambda item: item.score).tokens)
