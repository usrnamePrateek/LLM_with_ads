"""Repository classes for loading datasets, merging checkpoints, and saving generated JSON/CSV data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.ad_generation.config import CATEGORY_COLUMNS, QUERY_COLUMN
from src.ad_generation.entities import AdRecord


class QueryFrameRepository:
    """Repository for reading and checkpointing pandas DataFrames containing user queries."""

    def load(self, path: Path, query_column: str = QUERY_COLUMN) -> pd.DataFrame:
        """Loads a Parquet or CSV file containing queries, filtering out rows with missing query data."""
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported input type: {path}")

        if query_column not in df.columns:
            raise ValueError(f"Missing column {query_column!r} in {path}")

        df = df.dropna(subset=[query_column]).copy()
        print(f"Loaded {len(df):,} rows from {path}")
        return df

    def merge_checkpoint(self, checkpoint_path: Path, source_df: pd.DataFrame) -> pd.DataFrame:
        """Merges previously categorized query rows from a CSV checkpoint to resume interrupted batch generation jobs."""
        if not checkpoint_path.exists():
            return source_df

        resumed = pd.read_csv(checkpoint_path)
        print(f"Loaded checkpoint ({len(resumed):,} rows): {checkpoint_path}")
        out = source_df.copy()
        for col in CATEGORY_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA

        label_cols = [c for c in CATEGORY_COLUMNS if c in resumed.columns]
        if not label_cols:
            return out

        if "id" in out.columns and "id" in resumed.columns:
            ckpt = resumed[["id"] + label_cols].drop_duplicates(subset=["id"], keep="last")
            out = out.drop(columns=label_cols).merge(ckpt, on="id", how="left")
        else:
            n = min(len(out), len(resumed))
            for col in label_cols:
                out.iloc[:n, out.columns.get_loc(col)] = resumed[col].iloc[:n].to_numpy()
        return out

    def save_csv(self, frame: pd.DataFrame, path: Path) -> None:
        """Saves a pandas DataFrame to CSV, creating parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"  checkpoint saved: {path}")



class TextDomainRepository:
    """Loads a list of unique domain categories from a line-separated text file."""
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[str]:
        if not self._path.exists():
            raise FileNotFoundError(f"domains file not found: {self._path}")
        domains = [
            line.strip()
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not domains:
            raise ValueError(f"no domains found in {self._path}")
        return domains


class AdJsonlRepository:
    """Repository for persisting generated AdRecords as a JSONL file."""
    def __init__(self, path: Path) -> None:
        self._path = path

    def save(self, records: list[AdRecord]) -> None:
        """Saves a list of AdRecords into a JSONL file, ensuring ascii-safe unicode writing."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} ads to {self._path}")
