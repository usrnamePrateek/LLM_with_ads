from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryTopAd:
    query_id: str
    query: str
    ad_id: str
    ad_domain: str
    headline: str
    description: str
    cta: str
    score: float

    def to_dict(self) -> dict:
        return {
            "id": self.query_id,
            "query": self.query,
            "ad_id": self.ad_id,
            "ad_domain": self.ad_domain,
            "headline": self.headline,
            "description": self.description,
            "cta": self.cta,
            "score": self.score,
        }


@dataclass(frozen=True)
class PlacedAdResponse:
    query_id: str
    query: str
    ad_id: str
    ad_domain: str
    headline: str
    description: str
    cta: str
    position: str
    llm_response: str
    response_with_ad: str

    def to_dict(self) -> dict:
        return {
            "id": self.query_id,
            "query": self.query,
            "ad_id": self.ad_id,
            "ad_domain": self.ad_domain,
            "headline": self.headline,
            "description": self.description,
            "cta": self.cta,
            "position": self.position,
            "llm_response": self.llm_response,
            "response_with_ad": self.response_with_ad,
        }
