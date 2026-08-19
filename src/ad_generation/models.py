from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryCategory:
    domain: str
    intent: str
    commercial_intent: int

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "intent": self.intent,
            "commercial_intent": self.commercial_intent,
        }


@dataclass(frozen=True)
class AdCreative:
    headline: str
    description: str
    cta: str


@dataclass(frozen=True)
class AdRecord:
    domain: str
    ad_id: int
    headline: str
    description: str
    cta: str

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "ad_id": self.ad_id,
            "headline": self.headline,
            "description": self.description,
            "cta": self.cta,
        }
