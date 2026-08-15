from __future__ import annotations

from typing import Protocol

import numpy as np

from ad_retrieval.models import Ad, ScoredAd


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, ads: list[Ad]) -> None:
        ...

    def query(self, vector: np.ndarray, k: int) -> list[ScoredAd]:
        ...

    def save(self) -> None:
        ...

    def load(self) -> None:
        ...
