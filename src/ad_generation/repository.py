"""Repository classes for reading the taxonomy and appending generated ads to JSONL."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.ad_generation.entities import AdRecord


class AdJsonlRepository:
    """Repository for persisting generated AdRecords as a JSONL file."""
    def __init__(self, path: Path) -> None:
        self._path = path

    def get_existing_counts(self) -> dict[str, int]:
        """Reads the JSONL file and returns the number of ads generated per domain."""
        from collections import defaultdict
        counts = defaultdict(int)
        if not self._path.exists():
            return counts
            
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    domain = data.get("domain")
                    if domain:
                        counts[domain] += 1
                except json.JSONDecodeError:
                    continue
        return counts

    def save(self, records: list[AdRecord]) -> None:
        """Saves a list of AdRecords into a JSONL file, ensuring ascii-safe unicode writing."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} ads to {self._path}")

    def append(self, records: list[AdRecord]) -> None:
        """Appends a list of AdRecords to the JSONL file."""
        if not records:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        print(f"Appended {len(records)} ads to {self._path}")
class CsvTaxonomyRepository:
    """Loads taxonomy from CSV and calculates required ad counts per subtopic."""
    def __init__(self, path: Path, target_ads_per_topic: int = 1000) -> None:
        self._path = path
        self._target_ads = target_ads_per_topic

    def load_domain_counts(self) -> list[tuple[str, int, str, str]]:
        if not self._path.exists():
            raise FileNotFoundError(f"Taxonomy CSV not found: {self._path}")
        
        import pandas as pd
        df = pd.read_csv(self._path)
        if "Topic_Level_1" not in df.columns or "sub_topic" not in df.columns:
            raise ValueError(f"CSV missing required columns: {df.columns}")
            
        topic_groups = df.groupby("Topic_Level_1")["sub_topic"].apply(list).to_dict()
        
        results = []
        for topic, subtopics in topic_groups.items():
            n = len(subtopics)
            if n == 0:
                continue
            base_ads = self._target_ads // n
            remainder = self._target_ads % n
            
            for i, subtopic in enumerate(subtopics):
                count = base_ads
                if i == n - 1:
                    count += remainder
                
                domain_str = f"{topic} / {subtopic}"
                if count > 0:
                    results.append((domain_str, count, topic, subtopic))
                
        return results
