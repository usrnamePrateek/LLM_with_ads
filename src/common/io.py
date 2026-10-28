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


class CheckpointError(RuntimeError):
    """An existing output file cannot be used to resume, so the run stops instead of overwriting it."""


def read_checkpoint(path: Path, columns: Sequence[str], **read_csv_kwargs: Any) -> pd.DataFrame | None:
    """Reads an existing output CSV to resume from.

    Returns None when there is nothing to resume (file missing or 0 bytes). Raises CheckpointError when the file
    exists but cannot be read or lacks any of `columns`: starting fresh would overwrite it on the first save.
    """
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        existing = pd.read_csv(path, **read_csv_kwargs)
    except Exception as exc:
        raise CheckpointError(f"cannot read existing output {path} to resume ({exc}); fix or move it, then rerun") from exc
    missing = [column for column in columns if column not in existing.columns]
    if missing:
        raise CheckpointError(
            f"existing output {path} is missing columns {missing}, so it cannot be resumed; fix or move it, then rerun"
        )
    return existing


def load_done_keys(path: Path, columns: Sequence[str]) -> set[tuple[str, ...]] | None:
    """Returns the key tuples already written to a checkpoint CSV (read as str).

    Returns None when there is nothing to resume, so callers can tell "start fresh" apart from
    "resumed with zero rows". Raises CheckpointError for an unusable existing file (see read_checkpoint).
    """
    existing = read_checkpoint(path, columns, dtype=str)
    if existing is None:
        return None
    return set(zip(*(existing[column] for column in columns)))
