from __future__ import annotations

from dataclasses import dataclass


def make_ad_id(domain: str, ad_id: int) -> str:
    return f"{domain}::{ad_id}"


def make_embed_text(headline: str, description: str) -> str:
    return f"{headline.strip()}\n{description.strip()}"


@dataclass(frozen=True)
class Ad:
    id: str
    domain: str
    ad_id: int
    headline: str
    description: str
    cta: str
    embed_text: str

    @classmethod
    def from_record(cls, record: dict) -> Ad:
        domain = str(record["domain"]).strip()
        ad_id = int(record["ad_id"])
        headline = str(record["headline"]).strip()
        description = str(record["description"]).strip()
        cta = str(record.get("cta") or "").strip()
        if not domain or not headline or not description:
            raise ValueError("ad record missing domain, headline, or description")
        return cls(
            id=make_ad_id(domain, ad_id),
            domain=domain,
            ad_id=ad_id,
            headline=headline,
            description=description,
            cta=cta,
            embed_text=make_embed_text(headline, description),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "domain": self.domain,
            "ad_id": self.ad_id,
            "headline": self.headline,
            "description": self.description,
            "cta": self.cta,
            "embed_text": self.embed_text,
        }


@dataclass(frozen=True)
class ScoredAd:
    ad: Ad
    score: float
