"""Repositories for loading queries and saving synthetic ad placements."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ad_integration.config import QUERY_COLUMN
from src.common.io import load_queries, save_rows
from src.ad_integration.entities import PlacedAdResponse, QueryTopAd, SemanticParagraphScore


class QueryDatasetRepository:
    """Loads the base queries from either parquet or CSV."""
    def load(self, path: Path, query_column: str = QUERY_COLUMN) -> pd.DataFrame:
        return load_queries(path, query_column)


class AssignmentCsvRepository:
    """Persists the top-1 ad assignment for each query to a CSV."""
    def save(self, rows: list[QueryTopAd], path: Path, *, append: bool = False) -> None:
        save_rows(rows, path, append=append, label="rows")


class PlacedAdCsvRepository:
    """Loads ad assignments and saves the final response with the ad placed."""
    def save(
        self,
        rows: list[PlacedAdResponse],
        path: Path,
        *,
        append: bool = False,
    ) -> None:
        save_rows(rows, path, append=append, label="rows")

    def load_assignments(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"assignment csv not found: {path}")
        frame = pd.read_csv(path)
        required = ["id", "ad_id", "ad_domain", "headline", "description", "cta"]
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise ValueError(f"assignment csv missing columns {missing}")
        print(f"Loaded {len(frame):,} ad assignments from {path}")
        return frame


class SemanticParagraphCsvRepository:
    """Persists the detailed per-paragraph similarity scores for downstream analysis."""
    def save(
        self,
        rows: list[SemanticParagraphScore],
        path: Path,
        *,
        append: bool = False,
    ) -> None:
        save_rows(rows, path, append=append, label="paragraph rows")
