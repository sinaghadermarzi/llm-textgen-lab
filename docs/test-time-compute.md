# Test-time compute: history, terminology, and practice

Names in this area drifted faster than the ideas. This page fixes the vocabulary the lab uses, traces where each idea came from, and maps each one to importable code.

## A short genealogy

### Chain-of-thought prompting (2022)

Wei et al. (2022) prompted models with worked examples so that generation produces intermediate reasoning steps before the answer: one prompt, one visible reasoning trajectory, one answer. That is the original, narrow meaning of "chain of thought." It is a prompting technique, and its benefits were observed mainly on multi-step tasks and at sufficient model scale.

### Self-consistency (2022)

Two months later, Wang et al. (2022) observed that once reasoning is an explicitly generated sequence, sampling gives you as many trajectories as you can afford. Sample N trajectories independently, extract the final answer from each, and return the majority. The vote is over extracted answers, not full texts—different reasoning paths that reach the same answer count together. In the lab this is `orchestration.self_consistency` with `extract=extract_answer`, and `agreement_rate` reports how contested the vote was.

A useful way to read the method: the quantity of interest is `P(y|x) = Σ_r P(y|r, x) P(r|x)`, the answer marginalized over reasoning paths `r`. A single chain-of-thought trajectory picks one `r`; self-consistency approximates the sum by sampling several.

### Tree of Thoughts (2023)

Yao et al. (2023) generalized the single left-to-right trajectory into search: branch at intermediate states, evaluate partial progress, and abandon and back up when a branch looks bad. This needs more machinery than sampling complete trajectories—a state representation, a proposer, and an evaluator. In the lab, `search.dfs_backtrack` is the depth-first version (the `path.pop()` is the backtrack) and `search.beam_steps` is the breadth-first version.

### Reasoning models (2024–2026)

OpenAI's o1 (2024), DeepSeek-R1 (2025), Claude extended thinking, and Gemini thinking models moved the reasoning trajectory inside the model: reinforcement learning trains the model to produce a long internal chain that branches, checks itself, and backtracks in-context ("wait, let me reconsider") within a single call. The externally controllable quantity becomes effort or budget—`GenerationConfig.reasoning_effort` and `reasoning_budget` in this lab—rather than the decoding of individual steps. Even crude length control moves accuracy: s1 (2025) forces longer traces simply by suppressing the end of thinking and appending "Wait".

## How "chain-of-thought" became an umbrella term

Chain-of-thought and self-consistency arrived two months apart and were adopted as a package, so "we use chain-of-thought inference" often meant "generate reasoning, sample it several times, and aggregate." Today "CoT" can mean at least four things:

1. The 2022 prompting technique: worked examples that elicit visible steps.
2. Any visible intermediate reasoning text a model produces.
3. A provider's hidden or summarized internal reasoning stream.
4. The internal trace that reasoning-model training optimizes.

This lab uses "chain-of-thought" for generated intermediate reasoning (senses 1–2), "hidden reasoning" for sense 3 (see [chain-of-thought.md](chain-of-thought.md) for how the lab handles it safely), and "test-time compute" for the umbrella: any method that spends extra inference computation—more tokens, more calls, or search—to improve results.

## Voting is not search

The distinction that keeps this area legible:

- **Self-consistency and best-of-N sample complete trajectories independently** and aggregate only at the end. No trajectory ever influences another. (`self_consistency`, `best_of_n`)
- **Search couples trajectories at intermediate states**: branch, evaluate partial progress, abandon and back up. (`dfs_backtrack`, `beam_steps`, `mcts`)

Independent sampling parallelizes trivially and needs no state representation. Search can spend compute more selectively on deep problems, but only if partial progress can be evaluated.

## What practitioners actually use (early 2026)

1. **One call to a reasoning model with an effort or budget control.** The current default. Most branching and backtracking now happens inside the trace, so a single call replaces yesterday's external scaffolding for many tasks. Compare efforts in `01_api_controls.ipynb`.
2. **Parallel sampling plus voting.** Model cards still report cons@k—the majority vote over k samples—because it works and needs no verifier at inference time. Distinguish it from pass@k (did any of k samples succeed), which requires an oracle scorer and is therefore an evaluation number, not a deployable method. `self_consistency`; read `agreement_rate` as a confidence signal.
3. **Best-of-N with a verifier.** Generate candidates, score them with something external, keep the best. Outcome reward models score final answers; process reward models score individual steps (Lightman et al., 2023) and pair with `beam_steps`-shaped search. When the task has tests or a checker, those are the verifier and they are nearly free. `best_of_n` with a programmatic `score`.
4. **Execute-and-fix loops.** Generate, run the code or check the output, feed failures back, revise. This is the dominant deployed pattern for code, where the test suite is the verifier. `refine_loop` with `accept` as the checker.
5. **Iterative self-refinement.** Madaan et al. (2023) showed models can critique and improve their own output. The honest caveat: self-critique without an external signal is weak and can make answers worse; refinement earns its cost when the critique consumes tool or checker output.
6. **Adaptive budgets and cascades.** Answer cheap prompts cheaply; escalate to a bigger budget when the cheap attempt fails a check or when cheap samples disagree (low `agreement_rate`). `escalate` tries configurations cheapest-first.

Snell et al. (2024) study when these tokens are better spent at inference than on a bigger model. The answer is conditional on task difficulty and verifier quality, which matches this lab's insistence on plotting empirical curves rather than assuming a law.

## Where explicit tree search stands

Explicit Tree-of-Thoughts and MCTS over free-form text are today mostly research- and evaluation-side. RL-trained in-context backtracking replaced them for general use: the model does the branching internally, and one long call usually beats an orchestrated tree of short ones. Explicit search still earns its cost where a cheap, reliable external verifier exists—code against tests, puzzles against checkers, formal math against proof assistants. The lab includes `mcts`, `dfs_backtrack`, and `beam_steps` to make the mechanics inspectable, not as a deployment recommendation.

## Map from ideas to lab code

| Idea | Function | Notebook |
| --- | --- | --- |
| Self-consistency voting | `orchestration.self_consistency`, `majority_vote`, `agreement_rate` | 03 |
| Best-of-N with a verifier | `orchestration.best_of_n` | 03 |
| Execute-and-fix / refinement | `orchestration.refine_loop`, `critic_revision` | 03 |
| Model-as-judge selection | `orchestration.verifier_select` | 03 |
| Adaptive budget cascade | `orchestration.escalate` | 03 |
| DFS with backtracking (Tree of Thoughts) | `search.dfs_backtrack` | 04 |
| Stepwise beam (process-reward-guided shape) | `search.beam_steps` | 04 |
| Monte Carlo tree search | `search.mcts` | 04 |
| Effort and budget controls | `GenerationConfig.reasoning_effort` / `reasoning_budget` | 01 |

## References

- Wei et al., 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
- Wang et al., 2022. Self-Consistency Improves Chain of Thought Reasoning in Language Models.
- Yao et al., 2023. Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
- Madaan et al., 2023. Self-Refine: Iterative Refinement with Self-Feedback.
- Lightman et al., 2023. Let's Verify Step by Step.
- Snell et al., 2024. Scaling LLM Test-Time Compute Optimally Can Be More Effective Than Scaling Model Parameters.
- OpenAI, 2024. Learning to Reason with LLMs (o1).
- DeepSeek-AI, 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
- Muennighoff et al., 2025. s1: Simple Test-Time Scaling.
