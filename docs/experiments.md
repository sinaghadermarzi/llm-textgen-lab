# Experiment design notes

## Keep comparisons honest

- Fix the prompt set and scoring rule before sweeping a knob.
- Repeat stochastic conditions with several seeds.
- Report both quality and resource use.
- Compare at equal model, equal task, and approximately equal total budget.
- Separate visible output tokens from hidden/provider-reported reasoning tokens.
- Include failures, refusals, timeouts, and unsupported-parameter errors.

## A practical record

For each run, save:

- timestamp and repository commit
- provider/backend and model identifier
- prompt hash and prompt text when policy permits
- full generation configuration
- returned text and structured score
- input/output/reasoning token usage when available
- latency, number of calls, and estimated cost
- device and software versions for local runs

The project deliberately avoids claiming that a single curve establishes a law connecting compute and intelligence. Scaling curves are conditional on the model, task distribution, decoding algorithm, prompt, verifier, and metric.
