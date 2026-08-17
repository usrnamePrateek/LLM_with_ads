from __future__ import annotations

from pathlib import Path

import pandas as pd

from ad_retrieval.embeddings.base import EmbeddingEncoder
from ad_retrieval.store.base import VectorStore
from generation.config import AD_POSITIONS, ID_COLUMN, LLM_RESPONSE_COLUMN, QUERY_COLUMN, TOP_K
from generation.models import PlacedAdResponse, QueryTopAd
from generation.placement import format_ad_block, place_ad
from generation.repository import (
    AssignmentCsvRepository,
    PlacedAdCsvRepository,
    QueryDatasetRepository,
)


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


class PlaceAdsInResponsesService:
    def __init__(
        self,
        queries: QueryDatasetRepository,
        assignments: PlacedAdCsvRepository,
        writer: PlacedAdCsvRepository,
    ) -> None:
        self._queries = queries
        self._assignments = assignments
        self._writer = writer

    def run(
        self,
        dataset_path: Path,
        ads_csv_path: Path,
        output_path: Path,
        query_column: str = QUERY_COLUMN,
        limit: int | None = None,
    ) -> list[PlacedAdResponse]:
        dataset = self._queries.load(dataset_path, query_column=query_column)
        if LLM_RESPONSE_COLUMN not in dataset.columns:
            raise ValueError(f"dataset missing {LLM_RESPONSE_COLUMN!r}")
        ads = self._assignments.load_assignments(ads_csv_path)
        merged = dataset.merge(ads, on=ID_COLUMN, how="inner", suffixes=("", "_ad"))
        if query_column + "_ad" in merged.columns:
            merged = merged.drop(columns=[query_column + "_ad"])
        if limit is not None:
            merged = merged.head(limit).copy()
        print(f"Placing ads for {len(merged):,} queries ...")

        rows: list[PlacedAdResponse] = []
        for _, row in merged.iterrows():
            query_id = "" if pd.isna(row[ID_COLUMN]) else str(row[ID_COLUMN])
            query = "" if pd.isna(row[query_column]) else str(row[query_column])
            llm_response = "" if pd.isna(row[LLM_RESPONSE_COLUMN]) else str(row[LLM_RESPONSE_COLUMN])
            headline = "" if pd.isna(row["headline"]) else str(row["headline"])
            description = "" if pd.isna(row["description"]) else str(row["description"])
            cta = "" if pd.isna(row["cta"]) else str(row["cta"])
            ad_block = format_ad_block(headline, description, cta)
            for position in AD_POSITIONS:
                rows.append(
                    PlacedAdResponse(
                        query_id=query_id,
                        query=query,
                        ad_id="" if pd.isna(row["ad_id"]) else str(row["ad_id"]),
                        ad_domain="" if pd.isna(row["ad_domain"]) else str(row["ad_domain"]),
                        headline=headline,
                        description=description,
                        cta=cta,
                        position=position,
                        llm_response=llm_response,
                        response_with_ad=place_ad(llm_response, ad_block, position),
                    )
                )
        self._writer.save(rows, output_path)
        return rows
