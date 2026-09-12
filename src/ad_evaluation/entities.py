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
    """A scored preference comparing two ads for the same query."""
    query_id: str
    query: str
    ad_1_id: str
    ad_2_id: str
    winner_ad_id: str
    confidence: str
    is_swapped: bool


@dataclass(frozen=True)
class PlacementComparisonScore:
    """A scored preference comparing two different placements for the same ad and query."""
    query_id: str
    query: str
    ad_id: str
    pos_1: str
    pos_2: str
    winner_pos: str
    confidence: str
    is_swapped: bool
