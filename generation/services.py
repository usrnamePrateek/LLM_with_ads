from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_retrieval.embeddings.base import EmbeddingEncoder
from ad_retrieval.store.base import VectorStore
from generation.config import ID_COLUMN, QUERY_COLUMN, TOP_K
from generation.models import QueryTopAd
from generation.repository import AssignmentCsvRepository, QueryDatasetRepository


class AssignTopAdService:
    def __init__(
        self,
        queries: QueryDatasetRepository,
        encoder: EmbeddingEncoder,
        store: VectorStore,
        writer: AssignmentCsvRepository,
    ) -> None:
        self._queries = queries
        self._encoder = encoder
        self._store = store
        self._writer = writer

    def run(
        self,
        input_path: Path,
        output_path: Path,
        query_column: str = QUERY_COLUMN,
        k: int = TOP_K,
        limit: int | None = None,
    ) -> list[QueryTopAd]:
        frame = self._queries.load(input_path, query_column=query_column)
        if limit is not None:
            frame = frame.head(limit).copy()

        texts = frame[query_column].astype(str).tolist()
        print(f"Embedding {len(texts):,} queries ...")
        vectors = self._encoder.encode_queries(texts)

        assignments: list[QueryTopAd] = []
        for i, (_, row) in enumerate(frame.iterrows()):
            query = texts[i]
            query_id = "" if ID_COLUMN not in row.index or pd.isna(row[ID_COLUMN]) else str(row[ID_COLUMN])
            hits = self._store.query(vectors[i], k)
            if not hits:
                print(f"  no ad for query id={query_id!r}")
                continue
            hit = hits[0]
            ad = hit.ad
            assignments.append(
                QueryTopAd(
                    query_id=query_id,
                    query=query,
                    ad_id=ad.id,
                    ad_domain=ad.domain,
                    headline=ad.headline,
                    description=ad.description,
                    cta=ad.cta,
                    score=hit.score,
                )
            )
            if (i + 1) % 1000 == 0:
                print(f"  assigned {i + 1:,}/{len(texts):,}")

        self._writer.save(assignments, output_path)
        return assignments
