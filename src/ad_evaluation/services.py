"""Service classes for orchestrating the LLM-as-a-judge evaluation of ad placements."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.ad_evaluation.config import CHECKPOINT_EVERY, DEFAULT_SCORES_CSV, JUDGE_BATCH_SIZE
from src.ad_evaluation.llm.judge import VllmPlacementJudge
from src.ad_evaluation.core.masking import mask_ad_in_response
from src.ad_evaluation.entities import PlacementScore
from src.ad_evaluation.llm.parsing import parse_placement_score
from src.ad_evaluation.llm.prompts import build_user_prompt
from src.ad_evaluation.repository import PlacementInputRepository, PlacementScoreCsvRepository


class ScorePlacementsService:
    """Coordinates masking the raw ad, generating the LLM judge prompts, and parsing the scores."""
    def __init__(
        self,
        inputs: PlacementInputRepository,
        writer: PlacementScoreCsvRepository,
        judge: VllmPlacementJudge,
    ) -> None:
        self._inputs = inputs
        self._writer = writer
        self._judge = judge

    def run(
        self,
        positions_path: Path,
        ads_path: Path,
        output_path: Path = DEFAULT_SCORES_CSV,
        limit: int | None = None,
        batch_size: int = JUDGE_BATCH_SIZE,
    ) -> list[PlacementScore]:
        """Runs the evaluation pipeline over the dataset in batches."""
        frame = self._inputs.load(positions_path, ads_path)
        if limit is not None:
            frame = frame.head(limit).copy()
        wrote_any = False
        if output_path.exists():
            existing = pd.read_csv(output_path, dtype=str)
            scored_keys = set(
                zip(existing["id"], existing["ad_id"], existing["position"])
            )
            before = len(frame)
            frame = frame[
                ~frame.apply(
                    lambda r: (
                        "" if pd.isna(r["id"]) else str(r["id"]),
                        "" if pd.isna(r["ad_id"]) else str(r["ad_id"]),
                        "" if pd.isna(r["position"]) else str(r["position"]),
                    )
                    in scored_keys,
                    axis=1,
                )
            ].copy()
            wrote_any = True
            print(
                f"Resuming: {before - len(frame):,} already scored, "
                f"{len(frame):,} remaining."
            )

        print(f"Scoring {len(frame):,} placements ...")

        all_rows: list[PlacementScore] = []
        buffer: list[PlacementScore] = []
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            prompts: list[str] = []
            metas: list[tuple[str, str, str, str]] = []
            for _, row in batch.iterrows():
                query_id = "" if pd.isna(row["id"]) else str(row["id"])
                query = "" if pd.isna(row["query"]) else str(row["query"])
                ad_id = "" if pd.isna(row["ad_id"]) else str(row["ad_id"])
                position = "" if pd.isna(row["position"]) else str(row["position"])
                response = "" if pd.isna(row["response_with_ad"]) else str(row["response_with_ad"])
                headline = "" if pd.isna(row["headline"]) else str(row["headline"])
                description = "" if pd.isna(row["description"]) else str(row["description"])
                cta = "" if pd.isna(row["cta"]) else str(row["cta"])
                masked, ad_text = mask_ad_in_response(response, headline, description, cta)
                prompts.append(build_user_prompt(query, masked, ad_text))
                metas.append((query_id, query, ad_id, position))
            raw_texts = self._judge.generate(prompts)
            for meta, raw in zip(metas, raw_texts):
                query_id, query, ad_id, position = meta
                try:
                    score, reason = parse_placement_score(raw)
                except (ValueError, TypeError, KeyError) as exc:
                    print(f"  parse fail id={query_id!r} position={position!r}: {exc}")
                    score, reason = 0, f"parse_error: {exc}"
                buffer.append(
                    PlacementScore(
                        id=query_id,
                        query=query,
                        ad_id=ad_id,
                        position=position,
                        score=score,
                        reason=reason,
                    )
                )
            done = min(start + batch_size, len(frame))
            print(f"  scored {done:,}/{len(frame):,}")
            if len(buffer) >= CHECKPOINT_EVERY or done == len(frame):
                self._writer.save(buffer, output_path, append=wrote_any)
                wrote_any = True
                all_rows.extend(buffer)
                buffer = []
        return all_rows
