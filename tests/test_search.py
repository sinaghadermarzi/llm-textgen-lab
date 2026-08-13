import unittest

from llm_textgen_lab.search import beam_steps, dfs_backtrack, mcts


def propose_two_or_three(state):
    """Steps of +2 (preferred) or +3 toward 7; the +2-only path overshoots."""

    return [(state + 2, 0.6), (state + 3, 0.4)] if state < 7 else []


class SearchTests(unittest.TestCase):
    def test_mcts_expands_and_returns_child(self):
        def expand(state):
            return [(state + 1, 0.5), (state + 2, 0.5)] if state < 4 else []

        def evaluate(state):
            return 1.0 if state == 4 else -abs(4 - state)

        result = mcts(0, expand=expand, evaluate=evaluate, budget=30)
        self.assertIn(result.state, {1, 2})
        self.assertGreater(result.visits, 0)

    def test_dfs_backtrack_backtracks_to_goal(self):
        path = dfs_backtrack(0, expand=propose_two_or_three, is_goal=lambda state: state == 7, max_depth=5)
        self.assertEqual(path, [0, 2, 4, 7])

    def test_dfs_backtrack_prefers_higher_prior(self):
        def expand(state):
            return [(state + 1, 0.6), (state + 2, 0.4)] if state < 8 else []

        path = dfs_backtrack(0, expand=expand, is_goal=lambda state: state == 8, max_depth=8)
        self.assertEqual(path, [0, 1, 2, 3, 4, 5, 6, 7, 8])

    def test_dfs_backtrack_dead_end_everywhere_returns_none(self):
        self.assertIsNone(dfs_backtrack(0, expand=lambda state: [], is_goal=lambda state: state == 1))

    def test_dfs_backtrack_respects_max_depth(self):
        def expand(state):
            return [(state + 1, 1.0)] if state < 3 else []

        self.assertIsNone(dfs_backtrack(0, expand=expand, is_goal=lambda state: state == 3, max_depth=2))

    def test_dfs_backtrack_goal_root(self):
        path = dfs_backtrack(7, expand=propose_two_or_three, is_goal=lambda state: state == 7, max_depth=0)
        self.assertEqual(path, [7])

    def test_dfs_backtrack_prune_redirects_search(self):
        path = dfs_backtrack(
            0,
            expand=propose_two_or_three,
            is_goal=lambda state: state == 7,
            max_depth=5,
            prune=lambda state: state == 4,
        )
        self.assertEqual(path, [0, 2, 5, 7])

    def test_dfs_backtrack_negative_depth_raises(self):
        with self.assertRaises(ValueError):
            dfs_backtrack(0, expand=propose_two_or_three, is_goal=lambda state: state == 7, max_depth=-1)

    def test_beam_steps_finds_best_path(self):
        def expand(state):
            return [(state + 1, 0.5), (state + 2, 0.5)] if state < 8 else []

        def score(state):
            return 1.0 if state == 8 else -abs(8 - state) / 8

        path = beam_steps(0, expand=expand, score=score, width=2, depth=4)
        self.assertEqual(path, [0, 2, 4, 6, 8])

    def test_beam_steps_no_expansion_returns_root(self):
        path = beam_steps(0, expand=lambda state: [], score=lambda state: 0.0, width=2, depth=3)
        self.assertEqual(path, [0])

    def test_beam_steps_validates_width_and_depth(self):
        with self.assertRaises(ValueError):
            beam_steps(0, expand=lambda state: [], score=lambda state: 0.0, width=0)
        with self.assertRaises(ValueError):
            beam_steps(0, expand=lambda state: [], score=lambda state: 0.0, depth=0)

