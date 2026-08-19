from __future__ import annotations

from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompts: list[str]) -> list[str]:
        ...
