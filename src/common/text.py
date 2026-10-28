"""Text helpers for cleaning raw LLM output."""
from __future__ import annotations

import re


def strip_thinking(text: str) -> str:
    """Removes thinking tokens (e.g. <think>...</think>) often produced by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
