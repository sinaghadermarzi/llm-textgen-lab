"""Pure-Python sampling utilities for inspecting token selection mechanics.

The local Hugging Face track delegates production decoding to Transformers,
but these functions make the filtering steps visible without requiring torch.
"""

from __future__ import annotations

import math
import random
from typing import Iterable


def softmax(logits: Iterable[float], temperature: float = 1.0) -> list[float]:
    values = list(logits)
    if not values:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be positive for softmax")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exps = [math.exp(value - maximum) for value in scaled]
    total = sum(exps)
    return [value / total for value in exps]


def filter_logits(
    logits: Iterable[float],
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    min_p: float | None = None,
) -> list[float]:
    """Apply common filters and return renormalized probabilities.

    ``top_p`` is nucleus sampling. ``min_p`` retains tokens whose probability
    is at least ``min_p`` times the probability of the most likely token.
    Typical sampling is left to Transformers in the local backend because its
    implementation depends on version-specific logits processors.
    """

    probabilities = softmax(logits, temperature)
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        keep = set(sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)[:top_k])
        probabilities = [value if index in keep else 0.0 for index, value in enumerate(probabilities)]
    if min_p is not None:
        if not 0 < min_p <= 1:
            raise ValueError("min_p must be in (0, 1]")
        threshold = max(probabilities) * min_p if probabilities else 0.0
        probabilities = [value if value >= threshold else 0.0 for value in probabilities]
    if top_p is not None:
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        order = sorted(range(len(probabilities)), key=probabilities.__getitem__, reverse=True)
        keep: set[int] = set()
        cumulative = 0.0
        for index in order:
            keep.add(index)
            cumulative += probabilities[index]
            if cumulative >= top_p:
                break
        probabilities = [value if index in keep else 0.0 for index, value in enumerate(probabilities)]
    total = sum(probabilities)
    return [value / total for value in probabilities] if total else [1.0 / len(probabilities)] * len(probabilities)


def sample_index(probabilities: Iterable[float], *, rng: random.Random | None = None) -> int:
    values = list(probabilities)
    if not values or any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("probabilities must be non-negative and non-zero")
    chooser = rng or random
    return chooser.choices(range(len(values)), weights=values, k=1)[0]
