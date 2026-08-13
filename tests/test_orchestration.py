import unittest

from llm_textgen_lab.orchestration import (
    agreement_rate,
    best_of_n,
    critic_revision,
    escalate,
    extract_answer,
    majority_vote,
    refine_loop,
    self_consistency,
)
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


class ScriptedGenerator:
    """Returns queued texts in order; records prompts for inspection."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def __call__(self, prompt, config):
        self.calls.append(prompt)
        return GenerationResult(text=self.texts.pop(0), provider="fake", model="test")


class EffortGenerator:
    """Answers correctly only when the config asks for high reasoning effort."""

    def __call__(self, prompt, config):
        text = "8" if config.reasoning_effort == "high" else "7"
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

    def test_extract_answer_takes_text_after_marker(self):
        self.assertEqual(extract_answer("Work shown here.\nAnswer: 12"), "12")

    def test_extract_answer_uses_last_marker(self):
        self.assertEqual(extract_answer("Answer: 3\nWait, recompute.\nAnswer: 5\nDone."), "5")

    def test_extract_answer_falls_back_to_last_line(self):
        self.assertEqual(extract_answer("some thoughts\nfinal"), "final")

    def test_extract_answer_empty_text(self):
        self.assertEqual(extract_answer("   \n  "), "")

    def test_majority_vote_with_extract(self):
        candidates = [
            "Two plus two is four.\nAnswer: 4",
            "I count 2 + 2 = 4.\nAnswer: 4",
            "Answer: 5",
        ]
        self.assertEqual(majority_vote(candidates, extract=extract_answer), "4")

    def test_majority_vote_tie_prefers_first(self):
        self.assertEqual(majority_vote(["a", "b"]), "a")

    def test_agreement_rate(self):
        self.assertAlmostEqual(agreement_rate(["a", "a", "b"]), 2 / 3)

    def test_agreement_rate_empty_raises(self):
        with self.assertRaises(ValueError):
            agreement_rate([])

    def test_self_consistency_with_extract(self):
        generator = ScriptedGenerator(["r1\nAnswer: 7", "r2\nAnswer: 7", "r3\nAnswer: 9"])
        record = self_consistency(generator, "task", GenerationConfig(), count=3, extract=extract_answer)
        self.assertEqual(record.result.text, "7")
        self.assertAlmostEqual(record.result.metadata["agreement"], 2 / 3)
        self.assertEqual(record.calls, 3)

    def test_best_of_n_picks_highest_score(self):
        generator = ScriptedGenerator(["4", "8", "6"])
        record = best_of_n(generator, "task", GenerationConfig(), score=lambda text: -abs(8 - int(text)), count=3)
        self.assertEqual(record.result.text, "8")
        self.assertEqual(record.calls, 3)
        self.assertEqual(record.strategy, "best-of-n")

    def test_best_of_n_rejects_nonpositive_count(self):
        with self.assertRaises(ValueError):
            best_of_n(ScriptedGenerator([]), "task", GenerationConfig(), score=len, count=0)

    def test_refine_loop_accepts_draft(self):
        record = refine_loop(FakeGenerator(), "answer", GenerationConfig(), accept=lambda text: True)
        self.assertEqual(record.calls, 1)
        self.assertEqual(record.budget_label, "rounds=0")
        self.assertEqual(record.result.text, "42")

    def test_refine_loop_stops_when_revision_accepted(self):
        record = refine_loop(FakeGenerator(), "answer", GenerationConfig(), accept=lambda text: "revised" in text, max_rounds=3)
        self.assertEqual(record.calls, 3)
        self.assertEqual(record.result.text, "revised answer")
        self.assertEqual(record.strategy, "refine-loop")
        self.assertEqual(record.budget_label, "rounds=1")

    def test_refine_loop_exhausts_rounds(self):
        record = refine_loop(FakeGenerator(), "answer", GenerationConfig(), accept=lambda text: False, max_rounds=2)
        self.assertEqual(record.calls, 5)
        self.assertEqual(record.budget_label, "rounds=2")
        self.assertEqual(record.result.text, "revised answer")

    def test_refine_loop_zero_rounds_returns_draft(self):
        record = refine_loop(FakeGenerator(), "answer", GenerationConfig(), accept=lambda text: False, max_rounds=0)
        self.assertEqual(record.calls, 1)
        self.assertEqual(record.result.text, "42")

    def test_refine_loop_negative_rounds_raises(self):
        with self.assertRaises(ValueError):
            refine_loop(FakeGenerator(), "answer", GenerationConfig(), accept=lambda text: True, max_rounds=-1)

    def test_escalate_stops_at_first_accepted_tier(self):
        configs = [GenerationConfig(reasoning_effort="low"), GenerationConfig(reasoning_effort="high")]
        record = escalate(EffortGenerator(), "task", configs, accept=lambda text: text == "8")
        self.assertEqual(record.result.text, "8")
        self.assertEqual(record.calls, 2)
        self.assertEqual(record.budget_label, "tier-1")

    def test_escalate_accepts_first_tier(self):
        configs = [GenerationConfig(reasoning_effort="low"), GenerationConfig(reasoning_effort="high")]
        record = escalate(EffortGenerator(), "task", configs, accept=lambda text: True, labels=["cheap", "expensive"])
        self.assertEqual(record.calls, 1)
        self.assertEqual(record.budget_label, "cheap")

    def test_escalate_returns_last_attempt_when_none_accepted(self):
        configs = [GenerationConfig(), GenerationConfig(), GenerationConfig()]
        record = escalate(EffortGenerator(), "task", configs, accept=lambda text: False)
        self.assertEqual(record.calls, 3)
        self.assertEqual(record.result.text, "7")
        self.assertEqual(record.budget_label, "tier-2")

    def test_escalate_empty_configs_raises(self):
        with self.assertRaises(ValueError):
            escalate(EffortGenerator(), "task", [], accept=lambda text: True)

    def test_escalate_label_mismatch_raises(self):
        with self.assertRaises(ValueError):
            escalate(EffortGenerator(), "task", [GenerationConfig()], accept=lambda text: True, labels=["a", "b"])
