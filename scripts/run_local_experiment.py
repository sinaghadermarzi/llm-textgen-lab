#!/usr/bin/env python3
"""Run a local Hugging Face decoding experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_textgen_lab.local import HuggingFaceGenerator
from llm_textgen_lab.types import GenerationConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--prompt", default="Explain top-p sampling in one paragraph.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--typical-p", type=float)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--num-return-sequences", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--stop", action="append", default=[])
    args = parser.parse_args()
    generator = HuggingFaceGenerator(args.model, device=args.device)
    config = GenerationConfig(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        typical_p=args.typical_p,
        do_sample=args.beams == 1,
        num_beams=args.beams,
        num_return_sequences=args.num_return_sequences,
        repetition_penalty=args.repetition_penalty,
        stop_strings=tuple(args.stop),
    )
    result = generator.generate(args.prompt, config)
    print(result.text)
    print(f"\n[{result.model} on {result.metadata['device']}; {result.latency_ms:.0f} ms]")


if __name__ == "__main__":
    main()
