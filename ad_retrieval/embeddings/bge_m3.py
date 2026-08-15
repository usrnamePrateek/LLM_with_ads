from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ad_retrieval.config import EMBEDDING_MODEL_NAME, ENCODE_BATCH_SIZE


class BgeM3Encoder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = ENCODE_BATCH_SIZE,
        device: str | None = None,
    ) -> None:
        self._batch_size = batch_size
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SentenceTransformer(model_name, device=resolved_device)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > self._batch_size,
        )
        return np.asarray(vectors, dtype=np.float32)
