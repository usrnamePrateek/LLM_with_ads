"""Data classes defining domain objects for the ad placement pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTopAd:
    """Represents a single query matched with its best semantic ad."""
    id: str
    query: str
    ad_id: str
    ad_domain: str
    headline: str
    description: str
    cta: str
    score: float



@dataclass(frozen=True)
class PlacedAdResponse:
    """Represents a final LLM response with an ad synthetically inserted into it."""
    id: str
    query: str
    ad_id: str
    position: str
    response_with_ad: str



@dataclass(frozen=True)
class SemanticParagraphScore:
    """Records the semantic similarity score of a specific paragraph against an ad."""
    id: str
    query: str
    ad_id: str
    paragraph_index: int
    n_paragraphs: int
    paragraph: str
    cosine: float
    is_best: bool

