"""Data classes defining domain objects for the ad generation pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryCategory:
    """Represents the parsed categorization of a user query."""
    domain: str
    intent: str
    commercial_intent: int


@dataclass(frozen=True)
class AdCreative:
    """Represents the raw components of a generated ad before it is assigned an ID or domain."""
    headline: str
    description: str
    cta: str


@dataclass(frozen=True)
class AdRecord:
    """Represents a complete, serialized ad copy tied to a specific domain and ID."""
    domain: str
    ad_id: int
    headline: str
    description: str
    cta: str
    category: str | None = None
    subtopic: str | None = None