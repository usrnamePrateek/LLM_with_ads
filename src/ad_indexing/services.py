"""Service classes orchestrating the creation and querying of the ad vector index."""
from __future__ import annotations

from src.ad_indexing.embeddings.base import EmbeddingEncoder
from src.ad_indexing.entities import ScoredAd
from src.ad_indexing.repository import JsonlAdRepository
from src.ad_indexing.store.base import VectorStore


class IndexAdsService:
    """Coordinates loading ads, generating embeddings, and saving them to the vector store."""
    def __init__(
        self,
        repository: JsonlAdRepository,
        encoder: EmbeddingEncoder,
        store: VectorStore,
    ) -> None:
        self._repository = repository
        self._encoder = encoder
        self._store = store

    def run(self) -> int:
        """Executes the full indexing pipeline and persists the vector index to disk."""
        ads = self._repository.load()
        vectors = self._encoder.encode_documents([ad.embed_text for ad in ads])
        if vectors.shape[1] != self._encoder.dimension:
            raise ValueError("encoder dimension does not match produced vectors")
        self._store.upsert([ad.id for ad in ads], vectors, ads)
        self._store.save()
        return len(ads)


class SearchAdsService:
    """Coordinates embedding a user query and retrieving the most semantically similar ads."""
    def __init__(self, encoder: EmbeddingEncoder, store: VectorStore) -> None:
        self._encoder = encoder
        self._store = store

    def search(self, query: str, k: int) -> list[ScoredAd]:
        """Embeds a search query and queries the vector store for the top-k matches."""
        text = query.strip()
        if not text:
            raise ValueError("query must be non-empty")
        vectors = self._encoder.encode_queries([text])
        return self._store.query(vectors[0], k)
