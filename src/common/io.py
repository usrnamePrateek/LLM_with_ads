"""File I/O helpers shared by the pipeline modules: query loading, CSV row writing, and resume keys."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def load_queries(path: Path, query_column: str = "query") -> pd.DataFrame:
    """Loads a parquet or CSV query dataset, dropping rows with no query text."""
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


def save_rows(rows: Sequence[Any], path: Path, *, append: bool = False, label: str = "rows") -> None:
    """Writes dataclass rows to CSV; with append, adds to an existing file without repeating the header."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    if append and path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)
    print(f"Saved {len(rows):,} {label} to {path}")


def load_done_keys(path: Path, columns: Sequence[str]) -> set[tuple[str, ...]] | None:
    """Returns the key tuples already written to a checkpoint CSV (read as str).

    Returns None when the file is missing or lacks any of the key columns, so callers can tell
    "nothing to resume" apart from "resumed with zero rows".
    """
    if not path.exists():
        return None
    existing = pd.read_csv(path, dtype=str)
    if not all(column in existing.columns for column in columns):
        return None
    return set(zip(*(existing[column] for column in columns)))
