"""Data classes defining domain objects for the ad indexing and retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass


def make_ad_id(domain: str, ad_id: int) -> str:
    """Generates a globally unique identifier for an ad by combining its domain and local ad_id."""
    return f"{domain}::{ad_id}"


def make_embed_text(headline: str, description: str) -> str:
    """Formats the ad headline and description into a single string for vector embedding."""
    return f"{headline.strip()}\n{description.strip()}"


@dataclass(frozen=True)
class Ad:
    """Represents a fully processed ad ready for indexing, including its formatted embed_text."""
    id: str
    domain: str
    ad_id: int
    headline: str
    description: str
    cta: str
    embed_text: str

    @classmethod
    def from_record(cls, record: dict) -> Ad:
        """Parses a raw dictionary (e.g. from JSONL) into an Ad entity, generating IDs and embedding text."""
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



@dataclass(frozen=True)
class ScoredAd:
    """Represents an ad retrieved from the vector store alongside its similarity score."""
    ad: Ad
    score: float
