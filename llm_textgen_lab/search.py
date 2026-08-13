"""Small, model-agnostic search routines for inference labs: MCTS, DFS backtracking, and stepwise beam."""

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


def dfs_backtrack(
    root_state: State,
    *,
    expand: Callable[[State], Iterable[tuple[State, float]]],
    is_goal: Callable[[State], bool],
    max_depth: int = 8,
    prune: Callable[[State], bool] | None = None,
) -> list[State] | None:
    """Depth-first search that undoes steps on dead ends, Tree-of-Thoughts style.

    Children are tried in descending prior order: descend into the proposer's
    best-rated step first and back up to alternatives when a subtree fails.
    ``prune`` models an evaluator rejecting unpromising partial states.
    ``max_depth`` counts edges from the root and strictly bounds the search.
    Returns the root-to-goal path, or None when the bounded search is exhausted.
    """

    if max_depth < 0:
        raise ValueError("max_depth cannot be negative")
    path = [root_state]

    def descend(state: State, depth: int) -> bool:
        if is_goal(state):
            return True
        if depth == max_depth:
            return False
        for child, _prior in sorted(expand(state), key=lambda pair: -pair[1]):
            if prune is not None and prune(child):
                continue
            path.append(child)
            if descend(child, depth + 1):
                return True
            path.pop()  # the backtrack: undo the failed step and try the next branch
        return False

    return path if descend(root_state, 0) else None


def beam_steps(
    root_state: State,
    *,
    expand: Callable[[State], Iterable[tuple[State, float]]],
    score: Callable[[State], float],
    width: int = 3,
    depth: int = 4,
) -> list[State]:
    """Keep the best few partial paths per step, the shape process-reward-guided search takes.

    ``expand`` shares its contract with ``mcts`` and ``dfs_backtrack``; the
    proposal priors are accepted but ranking uses ``score`` on each path's last
    state. Returns the best final path; ties keep the earliest candidate.
    """

    if width <= 0 or depth <= 0:
        raise ValueError("width and depth must be positive")
    frontier = [[root_state]]
    for _ in range(depth):
        candidates = [path + [child] for path in frontier for child, _prior in expand(path[-1])]
        if not candidates:
            break
        frontier = sorted(candidates, key=lambda path: -score(path[-1]))[:width]
    return max(frontier, key=lambda path: score(path[-1]))
