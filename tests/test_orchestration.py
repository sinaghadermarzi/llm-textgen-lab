import unittest

from llm_textgen_lab.orchestration import critic_revision, majority_vote, self_consistency
from llm_textgen_lab.types import GenerationConfig, GenerationResult


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def __call__(self, prompt, config):
        self.calls.append(prompt)
        if "Revise" in prompt:
            text = "revised answer"
        elif "Critique" in prompt or "critique" in prompt.lower():
            text = "The candidate is correct."
        else:
            text = "42"
        return GenerationResult(text=text, provider="fake", model="test")


class OrchestrationTests(unittest.TestCase):
    def test_majority_vote(self):
        self.assertEqual(majority_vote(["a", "b", "a"]), "a")

    def test_self_consistency_counts_calls(self):
        generator = FakeGenerator()
        record = self_consistency(generator, "answer", GenerationConfig(), count=3)
        self.assertEqual(record.calls, 3)
        self.assertEqual(record.result.text, "42")

    def test_critic_revision_uses_three_calls(self):
        generator = FakeGenerator()
        record = critic_revision(generator, "answer", GenerationConfig())
        self.assertEqual(record.calls, 3)
        self.assertEqual(record.result.text, "revised answer")
