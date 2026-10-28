"""Repositories for persisting speed comparison records and summary statistics."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from src.common.io import save_rows
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
