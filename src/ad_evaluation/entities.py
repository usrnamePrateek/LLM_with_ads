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
