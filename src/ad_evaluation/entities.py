"""Data classes defining domain objects for placement evaluation."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementScore:
    """Represents a score and rationale assigned by the LLM judge for a specific ad placement."""
    id: str
    query: str
    ad_id: str
    position: str
    score: int
    reason: str


@dataclass(frozen=True)
class PairwisePreferenceScore:
    """Represents a preference chosen by the LLM between two ads for a given query."""
    query_id: str
    query: str
    ad_1_id: str
    ad_2_id: str
    winner_ad_id: str
    confidence: str
    is_swapped: bool
