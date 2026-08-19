from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlacementScore:
    query_id: str
    query: str
    ad_id: str
    position: str
    score: int
    reason: str

    def to_dict(self) -> dict:
        return {
            "id": self.query_id,
            "query": self.query,
            "ad_id": self.ad_id,
            "position": self.position,
            "score": self.score,
            "reason": self.reason,
        }
