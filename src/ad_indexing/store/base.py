from __future__ import annotations

from typing import Protocol

import numpy as np

from src.ad_indexing.models import Ad, ScoredAd


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, ads: list[Ad]) -> None:
        ...

    def query(self, vector: np.ndarray, k: int) -> list[ScoredAd]:
        ...

    def get_vector(self, ad_id: str) -> np.ndarray:
        ...

    def save(self) -> None:
        ...

    def load(self) -> None:
        ...
