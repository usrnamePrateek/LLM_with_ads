from __future__ import annotations

from src.ad_retrieval.embeddings.base import EmbeddingEncoder
from src.ad_retrieval.models import ScoredAd
from src.ad_retrieval.repository import AdRepository
from src.ad_retrieval.store.base import VectorStore


class IndexAdsService:
    def __init__(
        self,
        repository: AdRepository,
        encoder: EmbeddingEncoder,
        store: VectorStore,
    ) -> None:
        self._repository = repository
        self._encoder = encoder
        self._store = store

    def run(self) -> int:
        ads = self._repository.load()
        vectors = self._encoder.encode_documents([ad.embed_text for ad in ads])
        if vectors.shape[1] != self._encoder.dimension:
            raise ValueError("encoder dimension does not match produced vectors")
        self._store.upsert([ad.id for ad in ads], vectors, ads)
        self._store.save()
        return len(ads)


class SearchAdsService:
    def __init__(self, encoder: EmbeddingEncoder, store: VectorStore) -> None:
        self._encoder = encoder
        self._store = store

    def search(self, query: str, k: int) -> list[ScoredAd]:
        text = query.strip()
        if not text:
            raise ValueError("query must be non-empty")
        vectors = self._encoder.encode_queries([text])
        return self._store.query(vectors[0], k)
