"""Prompt helpers that keep reasoning experiments safe and measurable."""

SAFE_REASONING_SYSTEM = (
    "Solve the task carefully. Do not reveal private chain-of-thought. "
    "Return the answer, a concise rationale summary, and checkable evidence "
    "such as calculations, tests, or assumptions when useful."
)


def experiment_prompt(task: str, *, strategy: str = "direct") -> str:
    """Build a comparable prompt for direct, plan, or verify experiments."""

    instructions = {
        "direct": "Answer directly and state the key assumption if one matters.",
        "plan": "Before answering, make a short plan internally; show only a concise plan summary.",
        "verify": "Answer, then perform a private consistency check and report only corrections or evidence.",
        "structured": "Return a concise answer followed by a checklist of claims that can be verified externally.",
    }
    if strategy not in instructions:
        raise ValueError(f"unknown strategy: {strategy}")
    return f"{SAFE_REASONING_SYSTEM}\n\n{instructions[strategy]}\n\nTask:\n{task}"
