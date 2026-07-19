"""Small, model-agnostic MCTS implementation for search-based inference labs."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, TypeVar

State = TypeVar("State")


@dataclass
class SearchNode(Generic[State]):
    state: State
    parent: "SearchNode[State] | None" = None
    prior: float = 1.0
    children: list["SearchNode[State]"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def mcts(
    root_state: State,
    *,
    expand: Callable[[State], Iterable[tuple[State, float]]],
    evaluate: Callable[[State], float],
    budget: int = 100,
    exploration: float = 1.4,
    rng: random.Random | None = None,
) -> SearchNode[State]:
    """Run a compact UCT loop and return the most visited root child.

    The lab uses this to explain where tree search fits: a language model can
    propose continuations, while a verifier or environment supplies values.
    It is intentionally generic so it can be tested with arithmetic states.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    chooser = rng or random.Random(0)
    root = SearchNode(root_state)
    for _ in range(budget):
        node = root
        path = [node]
        while node.children:
            total = max(1, node.visits)
            node = max(node.children, key=lambda child: child.mean_value + exploration * child.prior * math.sqrt(math.log(total + 1) / (child.visits + 1)))
            path.append(node)
        expansions = list(expand(node.state))
        if expansions:
            node.children = [SearchNode(state, parent=node, prior=max(0.0, prior)) for state, prior in expansions]
            node = chooser.choice(node.children)
            path.append(node)
        value = evaluate(node.state)
        for visited in path:
            visited.visits += 1
            visited.value_sum += value
    return max(root.children, key=lambda child: child.visits) if root.children else root
