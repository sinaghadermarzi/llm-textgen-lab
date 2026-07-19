#!/usr/bin/env python3
"""Run one hosted-provider experiment, or print its planned request offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_textgen_lab.prompts import experiment_prompt
from llm_textgen_lab.providers import create_provider
from llm_textgen_lab.types import GenerationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--reasoning-budget", type=int)
    parser.add_argument("--prompt", default="Why does increasing inference-time compute sometimes improve accuracy?")
    parser.add_argument("--strategy", choices=["direct", "plan", "verify", "structured"], default="direct")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    defaults = {"openai": "gpt-5.6", "anthropic": "claude-sonnet-4-6", "gemini": "gemini-3.5-flash"}
    config = GenerationConfig(reasoning_effort=args.effort, reasoning_budget=args.reasoning_budget)
    prompt = experiment_prompt(args.prompt, strategy=args.strategy)
    plan = {"provider": args.provider, "model": args.model or defaults[args.provider], "config": config.__dict__, "prompt": prompt}
    if args.dry_run:
        print(json.dumps(plan, indent=2, default=list))
        return
    provider = create_provider(args.provider, args.model or defaults[args.provider])
    result = provider.generate(prompt, config)
    print(json.dumps({"text": result.text, "usage": result.usage, "latency_ms": result.latency_ms, "metadata": result.metadata}, indent=2, default=str))


if __name__ == "__main__":
    main()
