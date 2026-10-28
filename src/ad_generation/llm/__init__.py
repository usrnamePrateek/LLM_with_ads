from __future__ import annotations

from typing import TYPE_CHECKING
from src.ad_generation.llm.base import TextGenerator

if TYPE_CHECKING:
    from src.ad_generation.llm.ad_generator import VllmBulkAdGenerator

__all__ = ["TextGenerator", "VllmBulkAdGenerator"]


def __getattr__(name: str):
    if name == "VllmBulkAdGenerator":
        from src.ad_generation.llm.ad_generator import VllmBulkAdGenerator
        return VllmBulkAdGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
