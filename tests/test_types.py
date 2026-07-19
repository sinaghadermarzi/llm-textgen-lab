import unittest

from llm_textgen_lab.types import GenerationConfig


class GenerationConfigTests(unittest.TestCase):
    def test_transformers_mapping_omits_none(self):
        config = GenerationConfig(top_k=None, min_p=0.05, stop_strings=("END",))
        kwargs = config.to_transformers_kwargs()
        self.assertNotIn("top_k", kwargs)
        self.assertEqual(kwargs["min_p"], 0.05)
        self.assertEqual(kwargs["stop_strings"], ("END",))

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            GenerationConfig(top_p=0)

