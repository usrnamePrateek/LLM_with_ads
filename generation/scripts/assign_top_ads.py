from __future__ import annotations

import argparse
from pathlib import Path

from ad_retrieval.config import DEFAULT_INDEX_DIR
from ad_retrieval.embeddings.bge_m3 import BgeM3Encoder
from ad_retrieval.store.faiss_store import FaissVectorStore
from generation.config import DEFAULT_OUTPUT_CSV, DEFAULT_QUERIES_PARQUET, QUERY_COLUMN, TOP_K
from generation.repository import AssignmentCsvRepository, QueryDatasetRepository
from generation.services import AssignTopAdService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign the top-1 retrieved ad to each arena_preference_en_single_turn query."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_QUERIES_PARQUET)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--query-column", default=QUERY_COLUMN)
    parser.add_argument("--k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 1:
        raise SystemExit("--k must be >= 1")

    store = FaissVectorStore(args.index_dir)
    store.load()
    encoder = BgeM3Encoder()
    AssignTopAdService(
        queries=QueryDatasetRepository(),
        encoder=encoder,
        store=store,
        writer=AssignmentCsvRepository(),
    ).run(
        input_path=args.input,
        output_path=args.output,
        query_column=args.query_column,
        k=args.k,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
