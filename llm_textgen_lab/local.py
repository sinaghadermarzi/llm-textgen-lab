"""Hugging Face text generation backend with CUDA/MPS/CPU support."""

from __future__ import annotations

import time

from .device import detect_device
from .types import GenerationConfig, GenerationResult


class HuggingFaceGenerator:
    """Load a causal LM once and expose the lab's common generator interface."""

    provider = "huggingface"

    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the local extra to use HuggingFaceGenerator") from exc
        self.model_name = model_name
        self.device = detect_device(device)
        torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        import torch

        encoded = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        kwargs = config.to_transformers_kwargs()
        kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        started = time.perf_counter()
        with torch.inference_mode():
            try:
                output = self.model.generate(**encoded, **kwargs)
            except TypeError as exc:
                # ``min_p`` and ``stop_strings`` were added across different
                # Transformers versions. Retry without version-specific keys
                # and make the fallback visible in metadata.
                fallback = dict(kwargs)
                removed = []
                for key in ("min_p", "stop_strings", "seed"):
                    if key in fallback:
                        removed.append(key)
                        fallback.pop(key)
                output = self.model.generate(**encoded, **fallback)
                fallback_note = {"removed_unsupported_kwargs": removed, "error": str(exc)}
            else:
                fallback_note = {}
        prompt_tokens = encoded["input_ids"].shape[-1]
        texts = self.tokenizer.batch_decode(output[:, prompt_tokens:], skip_special_tokens=True)
        text = texts[0] if texts else ""
        for stop in config.stop_strings:
            if stop in text:
                text = text.split(stop, 1)[0]
        return GenerationResult(
            text=text,
            provider=self.provider,
            model=self.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            metadata={"device": self.device, "num_returned": len(texts), **fallback_note},
        )
