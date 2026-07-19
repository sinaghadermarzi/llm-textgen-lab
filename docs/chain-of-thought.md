# Chain-of-thought and hidden reasoning

This lab studies reasoning without treating a model's private chain-of-thought as a normal application output.

## Three different things

1. **Prompted rationale:** text the user asks the model to write. It can be useful, but it is a generated explanation and is not guaranteed to be a faithful transcript of the causal computation.
2. **Provider reasoning summaries:** some hosted APIs expose a short summary or usage count while keeping raw reasoning private or opaque.
3. **Local reasoning traces:** an open model may emit visible “thinking” tokens, special tags, or ordinary text. Visibility does not make those tokens a reliable explanation, and logging them can expose sensitive material.

The lab defaults to concise rationale summaries and checkable artifacts: equations, tests, citations, assumptions, or structured fields. The helper in `prompts.py` explicitly asks the model not to reveal private chain-of-thought.

## How to experiment safely

- Ask for an answer plus a short rationale summary, not a verbatim internal monologue.
- Prefer external verification: run tests, calculate with a trusted tool, or check a structured answer.
- Store usage and opaque metadata separately from user-visible text.
- Redact API keys, personal data, and model-generated secrets before saving transcripts.
- Treat summaries as evidence to inspect, not proof that the internal reasoning was faithful.
- When a provider returns encrypted or opaque reasoning state, pass it back only as documented; do not parse or rewrite it.

The educational goal is to understand compute allocation and decoding behavior. It does not require recovering private reasoning.
