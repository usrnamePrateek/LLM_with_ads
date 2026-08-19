from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_retrieval.embeddings.bge_m3 import BgeM3Encoder
from src.ad_retrieval.store.faiss_store import FaissVectorStore
from src.generation.config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_PLACED_ADS_CSV,
    DEFAULT_QUERIES_PARQUET,
    DEFAULT_SEMANTIC_PARAS_CSV,
    QUERY_COLUMN,
)
from src.generation.repository import (
    PlacedAdCsvRepository,
    QueryDatasetRepository,
    SemanticParagraphCsvRepository,
)
from src.generation.services import PlaceAdsInResponsesService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert the top-1 ad into each LMArena llm_response at first, middle, last, "
            "and semantic, and write per-chunk cosine scores."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_QUERIES_PARQUET)
    parser.add_argument("--ads", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLACED_ADS_CSV)
    parser.add_argument("--semantic-output", type=Path, default=DEFAULT_SEMANTIC_PARAS_CSV)
    parser.add_argument("--query-column", default=QUERY_COLUMN)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    writer = PlacedAdCsvRepository()
    encoder = BgeM3Encoder()
    store = FaissVectorStore(args.index_dir)
    store.load()
    PlaceAdsInResponsesService(
        queries=QueryDatasetRepository(),
        assignments=writer,
        writer=writer,
        semantic_writer=SemanticParagraphCsvRepository(),
        encoder=encoder,
        store=store,
    ).run(
        dataset_path=args.dataset,
        ads_csv_path=args.ads,
        output_path=args.output,
        semantic_output_path=args.semantic_output,
        query_column=args.query_column,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
