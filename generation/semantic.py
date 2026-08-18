from __future__ import annotations

import numpy as np

from ad_retrieval.embeddings.base import EmbeddingEncoder
from generation.splitting import join_chunks, split_paragraphs


def paragraph_cosine_scores(
    ad_vector: np.ndarray,
    paragraphs: list[str],
    encoder: EmbeddingEncoder,
) -> np.ndarray:
    if not paragraphs:
        return np.zeros((0,), dtype=np.float32)
    para_vecs = encoder.encode_documents(paragraphs)
    ad = np.asarray(ad_vector, dtype=np.float32).reshape(-1)
    return para_vecs @ ad


def score_chunks(
    llm_response: str,
    ad_vector: np.ndarray,
    encoder: EmbeddingEncoder,
) -> tuple[list[str], np.ndarray]:
    paragraphs = split_paragraphs(llm_response)
    if not paragraphs:
        return [], np.zeros((0,), dtype=np.float32)
    return paragraphs, paragraph_cosine_scores(ad_vector, paragraphs, encoder)


def insert_ad_after_best_paragraph(
    paragraphs: list[str],
    best_index: int,
) -> tuple[str, str]:
    """Insert after the best-scoring chunk."""
    if not paragraphs:
        return "", ""
    idx = min(max(best_index, 0), len(paragraphs) - 1)
    before = join_chunks(paragraphs[: idx + 1])
    after = join_chunks(paragraphs[idx + 1 :])
    return before, after
