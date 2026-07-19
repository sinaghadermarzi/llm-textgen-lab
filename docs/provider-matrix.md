# Provider control matrix

Provider interfaces are moving targets. This table describes the adapter contract used by the lab and links to the official documentation that should be checked before an experiment.

| Provider | Reasoning control in this lab | Common generation controls | Important caveat |
| --- | --- | --- | --- |
| OpenAI Responses API | `reasoning.effort` and optional `reasoning.summary` | `max_output_tokens`, optional stop/sampling when the model supports them | Supported effort values and sampling parameters vary by model; reasoning tokens are not raw visible text. |
| Anthropic Messages API | adaptive thinking with `output_config.effort`, or manual `budget_tokens` for compatible models | `max_tokens`; sampling is model/mode dependent | Adaptive thinking and manual budgets have different model support; thinking tokens count toward limits. |
| Google Gemini API | `thinking_level` for newer models, or `thinking_budget` for compatible earlier models | model-dependent `temperature`, `top_p`, `top_k`, output limits | Do not send `thinking_level` and `thinking_budget` together; model support and defaults vary. |
| Hugging Face / Transformers | user-controlled generation loop; no provider reasoning abstraction | temperature, top-k, top-p, min-p, typical-p, greedy, beam, repetition, stopping | Exact support depends on Transformers version and model implementation; the backend records fallbacks. |

Official references:

- [OpenAI reasoning models](https://developers.openai.com/api/docs/guides/reasoning)
- [Anthropic adaptive thinking](https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking)
- [Anthropic extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking)
- [Gemini thinking](https://ai.google.dev/gemini-api/docs/thinking)
- [Transformers generation strategies](https://huggingface.co/docs/transformers/main/en/generation_strategies)

The adapters intentionally expose a normalized `GenerationResult` but retain the actual request and raw usage metadata in `metadata`/`usage` so experiments can inspect provider differences.
