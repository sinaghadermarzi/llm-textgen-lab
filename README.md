# LLM Text Generation Lab

An educational, experiment-first lab for learning how language models generate text and how inference-time computation changes results.

The lab has two complementary tracks:

1. **Hosted APIs** — compare provider-exposed sampling and reasoning controls without running a model locally.
2. **Local Hugging Face models** — inspect and change the decoding loop with PyTorch on CUDA, Apple Metal/MPS, or CPU.

The repository is intentionally usable in layers. The core package has no mandatory third-party dependencies, offline tests use deterministic fakes, and model/API extras are opt-in.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

To run a hosted experiment, install one or more provider extras and export only the key you need:

```bash
python -m pip install -e '.[api]'
cp .env.example .env
export OPENAI_API_KEY='…'       # or ANTHROPIC_API_KEY / GEMINI_API_KEY
python scripts/run_api_experiment.py --provider openai --model gpt-5.6 --effort low
```

To run a local experiment, install the local extra. The first run downloads the model named by `--model` from Hugging Face:

```bash
python -m pip install -e '.[local]'
python scripts/run_local_experiment.py --model Qwen/Qwen3-0.6B --device auto --prompt 'Explain top-p sampling.'
```

`--device auto` prefers CUDA, then Apple MPS, then CPU. Override it with `cuda`, `mps`, or `cpu` when comparing backends.

## Curriculum

Read [docs/curriculum.md](docs/curriculum.md) in order. The short version is:

- `notebooks/01_api_controls.ipynb` — provider reasoning controls, sampling controls, usage, latency, and cost.
- `notebooks/02_local_decoding.ipynb` — logits, greedy decoding, sampling, beam search, stopping, and repetition controls.
- `notebooks/03_test_time_compute.ipynb` — multiple candidates, self-consistency, verifier/critic loops, adaptive budgets, and search.
- `notebooks/04_mcts_search.ipynb` — a model-agnostic Monte Carlo tree search exercise.

The notebooks are companion material; every important operation also exists in importable modules and scripts so it can be tested and automated.

## What this lab measures

For each experiment, record correctness or task score, visible output length, provider-reported reasoning/thinking tokens when available, wall-clock latency, number of calls, and estimated cost. Compare configurations at equal budgets where possible. Treat `temperature`, `top_p`, and `top_k` as decoding controls—not as universal substitutes for reasoning effort.

The lab treats hidden chain-of-thought carefully. It does not ask hosted providers to reveal raw private reasoning. Instead, use provider-supported summaries or opaque reasoning state, ask for concise answers plus checkable artifacts, and study externally observable behavior. See [docs/chain-of-thought.md](docs/chain-of-thought.md).

## Project layout

```text
llm_textgen_lab/       dependency-light core, providers, local generation, search
configs/               reproducible experiment presets
docs/                  explanations, safety notes, provider matrix, curriculum
notebooks/             guided lab notebooks
scripts/               command-line entry points
tests/                 offline unit tests
```

## Provider notes

Provider APIs change. The adapters keep the common interface small and pass provider-specific options through `provider_options`. Check the current provider documentation before changing model names or relying on a newly introduced parameter; links are maintained in [docs/provider-matrix.md](docs/provider-matrix.md).

## License

This repository is an educational lab. Add a license before redistributing it as a library or dataset.
