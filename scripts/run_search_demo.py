#!/usr/bin/env python3
"""Run a tiny MCTS problem to make search-based inference concrete."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_textgen_lab.search import mcts


def expand(state: int):
    return [(state + 1, 0.6), (state + 2, 0.4)] if state < 8 else []


def evaluate(state: int) -> float:
    return 1.0 if state == 8 else -abs(8 - state) / 8


if __name__ == "__main__":
    result = mcts(0, expand=expand, evaluate=evaluate, budget=100)
    print({"best_first_step": result.state, "visits": result.visits, "value": result.mean_value})
