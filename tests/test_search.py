import unittest

from llm_textgen_lab.search import mcts


class SearchTests(unittest.TestCase):
    def test_mcts_expands_and_returns_child(self):
        def expand(state):
            return [(state + 1, 0.5), (state + 2, 0.5)] if state < 4 else []

        def evaluate(state):
            return 1.0 if state == 4 else -abs(4 - state)

        result = mcts(0, expand=expand, evaluate=evaluate, budget=30)
        self.assertIn(result.state, {1, 2})
        self.assertGreater(result.visits, 0)

