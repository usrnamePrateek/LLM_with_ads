from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.generation.config import QUERY_COLUMN
from src.generation.models import PlacedAdResponse, QueryTopAd, SemanticParagraphScore


class QueryDatasetRepository:
    def load(self, path: Path, query_column: str = QUERY_COLUMN) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(path)
        elif suffix == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError(f"unsupported dataset type: {path}")
        if query_column not in frame.columns:
            raise ValueError(f"missing column {query_column!r} in {path}")
        frame = frame.dropna(subset=[query_column]).copy()
        print(f"Loaded {len(frame):,} queries from {path}")
        return frame


class AssignmentCsvRepository:
    def save(self, rows: list[QueryTopAd], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([row.to_dict() for row in rows])
        frame.to_csv(path, index=False)
        print(f"Saved {len(rows):,} rows to {path}")


class PlacedAdCsvRepository:
    def save(
        self,
        rows: list[PlacedAdResponse],
        path: Path,
        *,
        append: bool = False,
    ) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([row.to_dict() for row in rows])
        if append and path.exists():
            frame.to_csv(path, mode="a", header=False, index=False)
        else:
            frame.to_csv(path, index=False)
        print(f"Saved {len(rows):,} rows to {path}")

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
    def save(
        self,
        rows: list[SemanticParagraphScore],
        path: Path,
        *,
        append: bool = False,
    ) -> None:
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([row.to_dict() for row in rows])
        if append and path.exists():
            frame.to_csv(path, mode="a", header=False, index=False)
        else:
            frame.to_csv(path, index=False)
        print(f"Saved {len(rows):,} paragraph rows to {path}")
