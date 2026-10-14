"""Repositories for persisting speed comparison records and summary statistics."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

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
        if not records:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([asdict(record) for record in records])
        if append and path.exists():
            frame.to_csv(path, mode="a", header=False, index=False)
        else:
            frame.to_csv(path, index=False)
        print(f"Saved {len(records):,} records to {path}")

    def load_processed_ids(self, path: Path) -> set[str]:
        """Loads already processed query IDs from an existing output CSV for checkpointing."""
        if not path.exists():
            return set()
        try:
            frame = pd.read_csv(path)
            if ID_COLUMN in frame.columns:
                processed = set(frame[ID_COLUMN].astype(str))
                print(f"Found existing output at {path} with {len(processed):,} processed IDs.")
                return processed
        except Exception as err:
            print(f"Warning: could not read existing output for checkpointing: {err}")
        return set()

    def save_summary(self, summary: SpeedBenchmarkSummary, path: Path) -> None:
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
