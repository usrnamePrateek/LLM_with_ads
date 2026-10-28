from __future__ import annotations

from typing import TYPE_CHECKING
from src.ad_generation.llm.base import TextGenerator

if TYPE_CHECKING:
    from src.ad_generation.llm.ad_generator import VllmBulkAdGenerator
    from src.ad_generation.llm.category_generator import VllmCategoryGenerator

__all__ = ["TextGenerator", "VllmBulkAdGenerator", "VllmCategoryGenerator"]


def __getattr__(name: str):
    if name == "VllmBulkAdGenerator":
        from src.ad_generation.llm.ad_generator import VllmBulkAdGenerator
        return VllmBulkAdGenerator
    if name == "VllmCategoryGenerator":
        from src.ad_generation.llm.category_generator import VllmCategoryGenerator
        return VllmCategoryGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
