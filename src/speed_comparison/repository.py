"""Repositories for persisting speed comparison records and summary statistics."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.common.io import read_checkpoint, save_rows
from src.speed_comparison.config import ID_COLUMN
from src.speed_comparison.entities import QuerySpeedComparisonRecord, SpeedBenchmarkSummary


class SpeedComparisonRepository:
    """Manages CSV persistence for query records and JSON export for summary metrics."""

    def save_records(
        self,
        records: Sequence[QuerySpeedComparisonRecord],
        path: Path,
        *,
        append: bool = False,
    ) -> None:
        """Persists comparison records to CSV using dataclasses.asdict."""
        save_rows(records, path, append=append, label="records")

    def save_retrieval_records(
        self,
        records: Sequence[object],
        path: Path,
    ) -> None:
        """Persists retrieval-only timing records to CSV using dataclasses.asdict."""
        save_rows(records, path, label="retrieval records")

    def load_processed_ids(self, path: Path) -> set[str]:
        """Loads already processed query IDs from an existing output CSV for checkpointing."""
        frame = read_checkpoint(path, [ID_COLUMN])
        if frame is None:
            return set()
        processed = set(frame[ID_COLUMN].astype(str))
        print(f"Found existing output at {path} with {len(processed):,} processed IDs.")
        return processed

    def save_summary(
        self,
        summary: SpeedBenchmarkSummary,
        path: Path,
    ) -> None:
        """Persists the aggregated benchmark summary to a structured JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(summary)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved benchmark summary to {path}")

    def load_records(self, path: Path) -> pd.DataFrame:
        """Loads existing benchmark records from CSV into a pandas DataFrame."""
        if not path.exists():
            raise FileNotFoundError(f"records file not found: {path}")
        return pd.read_csv(path)
