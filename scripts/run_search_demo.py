#!/usr/bin/env python3
"""Run tiny search problems (MCTS, DFS backtracking, stepwise beam) to make search-based inference concrete."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_textgen_lab.search import beam_steps, dfs_backtrack, mcts


def expand(state: int):
    return [(state + 1, 0.6), (state + 2, 0.4)] if state < 8 else []


def evaluate(state: int) -> float:
    return 1.0 if state == 8 else -abs(8 - state) / 8


def propose(state: int):
    """Steps of +2 (preferred) or +3 toward 7; the +2-only path overshoots, forcing a backtrack."""

    return [(state + 2, 0.6), (state + 3, 0.4)] if state < 7 else []


if __name__ == "__main__":
    result = mcts(0, expand=expand, evaluate=evaluate, budget=100)
    print({"best_first_step": result.state, "visits": result.visits, "value": result.mean_value})
    print({"dfs_path": dfs_backtrack(0, expand=propose, is_goal=lambda state: state == 7, max_depth=5)})
    print({"beam_path": beam_steps(0, expand=expand, score=evaluate, width=2, depth=4)})
