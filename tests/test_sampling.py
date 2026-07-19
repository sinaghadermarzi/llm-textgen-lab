import random
import unittest

from llm_textgen_lab.sampling import filter_logits, sample_index, softmax


class SamplingTests(unittest.TestCase):
    def test_softmax_sums_to_one(self):
        self.assertAlmostEqual(sum(softmax([1.0, 2.0, 3.0])), 1.0)

    def test_top_k_zeroes_other_tokens(self):
        probabilities = filter_logits([4.0, 3.0, 2.0], top_k=1)
        self.assertEqual(probabilities.count(0.0), 2)
        self.assertEqual(probabilities[0], 1.0)

    def test_sampling_is_reproducible_with_rng(self):
        rng = random.Random(7)
        self.assertEqual(sample_index([0.1, 0.9], rng=rng), 1)

