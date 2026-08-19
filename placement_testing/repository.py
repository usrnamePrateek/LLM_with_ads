from __future__ import annotations

from pathlib import Path

import pandas as pd

from placement_testing.models import PlacementScore


class PlacementInputRepository:
    def load(self, positions_path: Path, ads_path: Path) -> pd.DataFrame:
        if not positions_path.exists():
            raise FileNotFoundError(f"positions csv not found: {positions_path}")
        if not ads_path.exists():
            raise FileNotFoundError(f"ads csv not found: {ads_path}")
        positions = pd.read_csv(positions_path)
        required_pos = ["id", "query", "ad_id", "position", "response_with_ad"]
        missing_pos = [col for col in required_pos if col not in positions.columns]
        if missing_pos:
            raise ValueError(f"positions csv missing columns {missing_pos}")
        ads = pd.read_csv(ads_path)
        required_ads = ["id", "headline", "description", "cta"]
        missing_ads = [col for col in required_ads if col not in ads.columns]
        if missing_ads:
            raise ValueError(f"ads csv missing columns {missing_ads}")
        ads = ads[required_ads].drop_duplicates(subset=["id"])
        merged = positions.merge(ads, on="id", how="left", suffixes=("", "_ad"))
        print(f"Loaded {len(merged):,} placed responses from {positions_path}")
        return merged


class PlacementScoreCsvRepository:
    def save(
        self,
        rows: list[PlacementScore],
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
        print(f"Saved {len(rows):,} scores to {path}")
