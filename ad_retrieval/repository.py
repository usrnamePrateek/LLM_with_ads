from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ad_retrieval.models import Ad


class AdRepository(Protocol):
    def load(self) -> list[Ad]:
        ...


class JsonlAdRepository:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[Ad]:
        if not self._path.exists():
            raise FileNotFoundError(f"ads jsonl not found: {self._path}")

        ads: list[Ad] = []
        seen: set[str] = set()
        with self._path.open(encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_no} of {self._path}") from exc
                ad = Ad.from_record(record)
                if ad.id in seen:
                    raise ValueError(f"duplicate ad id {ad.id!r} on line {line_no}")
                seen.add(ad.id)
                ads.append(ad)
        if not ads:
            raise ValueError(f"no ads found in {self._path}")
        return ads
