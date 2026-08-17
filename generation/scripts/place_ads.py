from __future__ import annotations

import argparse
from pathlib import Path

from generation.config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_PLACED_ADS_CSV,
    DEFAULT_QUERIES_PARQUET,
    QUERY_COLUMN,
)
from generation.repository import PlacedAdCsvRepository, QueryDatasetRepository
from generation.services import PlaceAdsInResponsesService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Insert the top-1 ad into each LMArena llm_response at first, middle, and last."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_QUERIES_PARQUET)
    parser.add_argument("--ads", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLACED_ADS_CSV)
    parser.add_argument("--query-column", default=QUERY_COLUMN)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    writer = PlacedAdCsvRepository()
    PlaceAdsInResponsesService(
        queries=QueryDatasetRepository(),
        assignments=writer,
        writer=writer,
    ).run(
        dataset_path=args.dataset,
        ads_csv_path=args.ads,
        output_path=args.output,
        query_column=args.query_column,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
