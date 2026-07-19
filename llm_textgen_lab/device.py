"""Device detection that works when PyTorch is not installed."""

from __future__ import annotations

import os


def detect_device(preference: str = "auto") -> str:
    """Return ``cuda``, ``mps``, or ``cpu``.

    ``auto`` prefers CUDA for throughput, then Apple Metal/MPS, then CPU.
    Explicit choices fail fast when the requested accelerator is unavailable.
    """

    requested = os.getenv("LLM_TEXTGEN_DEVICE", preference).lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("device must be one of auto, cuda, mps, cpu")
    try:
        import torch
    except ImportError:
        if requested != "auto" and requested != "cpu":
            raise RuntimeError("PyTorch is required for accelerator detection")
        return "cpu"

    available = {
        "cuda": bool(torch.cuda.is_available()),
        "mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    }
    if requested == "auto":
        return "cuda" if available["cuda"] else "mps" if available["mps"] else "cpu"
    if requested != "cpu" and not available[requested]:
        raise RuntimeError(f"Requested device {requested!r} is not available")
    return requested
