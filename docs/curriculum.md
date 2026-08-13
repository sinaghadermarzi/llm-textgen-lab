# Curriculum

The lab is organized around a useful distinction:

- **Model controls** change how one model call generates tokens.
- **Orchestration controls** spend more calls or arrange calls into a search, voting, critique, or verification procedure.
- **Provider reasoning controls** ask a hosted reasoning model to allocate internal computation, but the exact decoder remains provider-controlled.

## 1. Establish a baseline

Start with the same prompt set and a deterministic or low-temperature baseline. Include easy factual prompts, short-answer arithmetic, a multi-step word problem, a code task with tests, and an ambiguous planning task. Save the prompt, configuration, result, usage, latency, and score for every run.

## 2. Hosted API track

In `01_api_controls.ipynb`:

1. Run the same prompt at several provider reasoning efforts or thinking levels.
2. Compare provider-reported internal reasoning/thinking usage when exposed.
3. Hold the model and prompt fixed while varying visible output limits.
4. Test which sampling knobs are accepted for each model; do not assume all providers support temperature, top-p, or top-k on reasoning models.
5. Compare one high-effort call with several lower-effort calls at a similar estimated budget.

The important result is an empirical curve, not a universal “intelligence per token” law. Plot quality against total calls, billed tokens, visible output tokens, and latency separately.

## 3. Local decoding track

In `02_local_decoding.ipynb`:

1. Load a small causal language model with `HuggingFaceGenerator`.
2. Confirm the selected backend: CUDA, MPS, or CPU.
3. Compare greedy decoding with sampling.
4. Sweep temperature, top-k, top-p, min-p, and typical-p where supported.
5. Compare beam width and number of returned sequences.
6. Add repetition penalties, no-repeat n-gram constraints, and stop strings.
7. Inspect the pure-Python `sampling.py` functions to see why filtering changes the probability mass.

Some knobs are version- or model-dependent. The local backend records unsupported options in result metadata rather than silently claiming they were applied.

## 4. Orchestration and test-time compute

In `03_test_time_compute.ipynb`:

- **Self-consistency:** generate independent candidates, vote on the extracted final answer (`majority_vote` with `extract=`), and read the agreement rate as a confidence signal.
- **Best-of-N:** generate candidates and keep the one a programmatic scorer—tests, a checker, a reward-model stand-in—likes best (`best_of_n`).
- **Draft–critic–revision:** spend three calls to create, critique, and revise an answer; `refine_loop` iterates the same cycle until an external check accepts.
- **Verifier selection:** generate candidates, then ask a verifier to score them.
- **Adaptive allocation:** try configurations cheapest-first and stop at the first accepted answer (`escalate`); low agreement among cheap samples is a natural escalation trigger.
- **Search:** use a language model as a proposal function and a verifier, environment, or test suite as the value function.

Always compare against a one-call baseline and count the complete budget, including verifier calls. For where these methods came from and which ones practitioners actually use, read [test-time-compute.md](test-time-compute.md).

## 5. Tree search: MCTS, DFS backtracking, and stepwise beam

`04_mcts_search.ipynb` and `scripts/run_search_demo.py` use a tiny arithmetic state space. `dfs_backtrack` makes the Tree-of-Thoughts move concrete—descend into the preferred step and undo it when the subtree fails—while `beam_steps` keeps the best few partial paths per step, the frontier shape used with process reward models. Modern reasoning models internalize the same backtracking inside one long trace (see [test-time-compute.md](test-time-compute.md)), which is why explicit search over free-form text is now mostly research-side.

The point is not that MCTS is automatically useful for free-form text. The point is to identify the missing ingredients needed to make search useful: a way to propose continuations, a state representation, a value function, and a stopping rule. For code, tests can provide a value signal; for puzzles, a checker can; for open-ended writing, automated value is much harder.

## Suggested research questions

1. Does more reasoning effort help easy prompts, or only hard prompts?
2. At equal cost, is one high-effort call better than multiple low-effort candidates?
3. How much does a verifier improve accuracy when it shares the same model as the solver?
4. Where do additional candidates show diminishing returns?
5. Does higher temperature improve diversity enough to outweigh lower per-sample accuracy?
6. Do beam search and sampling fail in different ways on open-ended text?
7. How sensitive are results to prompt wording and scoring rules?
