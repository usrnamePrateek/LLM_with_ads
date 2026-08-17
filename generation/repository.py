from __future__ import annotations

from pathlib import Path

import pandas as pd

from generation.config import QUERY_COLUMN
from generation.models import QueryTopAd


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
