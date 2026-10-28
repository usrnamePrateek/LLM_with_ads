from __future__ import annotations

from typing import TYPE_CHECKING
from src.ad_generation.llm.base import TextGenerator

if TYPE_CHECKING:
    from src.ad_generation.llm.ad_generator import VllmAdGenerator
    from src.ad_generation.llm.category_generator import VllmCategoryGenerator

__all__ = ["TextGenerator", "VllmAdGenerator", "VllmCategoryGenerator"]


def __getattr__(name: str):
    if name == "VllmAdGenerator":
        from src.ad_generation.llm.ad_generator import VllmAdGenerator
        return VllmAdGenerator
    if name == "VllmCategoryGenerator":
        from src.ad_generation.llm.category_generator import VllmCategoryGenerator
        return VllmCategoryGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
