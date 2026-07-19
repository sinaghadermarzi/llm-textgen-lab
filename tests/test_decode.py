import unittest

from llm_textgen_lab.decode import beam_search, greedy_decode


def logits(sequence):
    return [0.0, 2.0, 1.0] if len(sequence) % 2 else [0.0, 1.0, 3.0]


class DecodeTests(unittest.TestCase):
    def test_greedy_selects_local_maximum(self):
        self.assertEqual(greedy_decode(logits, steps=2), [0, 1, 2])

    def test_beam_search_returns_requested_shape(self):
        self.assertEqual(len(beam_search(logits, steps=3, width=2)), 4)
