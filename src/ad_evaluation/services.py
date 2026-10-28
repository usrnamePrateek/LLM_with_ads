"""Service classes for orchestrating the LLM-as-a-judge evaluation of ad placements."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Literal

import pandas as pd

from src.ad_evaluation.config import CHECKPOINT_EVERY, DEFAULT_SCORES_CSV, JUDGE_BATCH_SIZE
from src.ad_evaluation.core.masking import AdBlockNotFoundError, mask_ad_in_response
from src.ad_evaluation.entities import PairwisePreferenceScore, PlacementComparisonScore, PlacementScore
from src.ad_evaluation.llm.parsing import (
    parse_pairwise_preference,
    parse_placement_comparison,
    parse_placement_score,
)
from src.ad_evaluation.llm.prompts import (
    build_pairwise_preference_prompt,
    build_placement_comparison_prompt,
    build_user_prompt,
)
from src.ad_evaluation.repository import (
    PlacementComparisonScoreCsvRepository,
    PlacementInputRepository,
    PlacementScoreCsvRepository,
    PreferenceScoreCsvRepository,
)
from src.ad_indexing.entities import Ad
from src.ad_indexing.repository import JsonlAdRepository
from src.common.io import load_done_keys

if TYPE_CHECKING:
    from src.ad_evaluation.llm.judge import VllmPlacementJudge
    from src.ad_evaluation.llm.preference_judge import VllmPlacementComparisonJudge, VllmPreferenceJudge

PreferenceCandidates = Literal["same_category", "cross_category"]


def _report_unparsed(count: int) -> None:
    """Judge answers that failed to parse are not saved, so a resumed run asks the judge again."""
    if count:
        print(f"{count:,} judge answers failed to parse (not saved; retried on resume).")


def _cell(value: object) -> str:
    """Reads a CSV cell the way place_ads wrote it: missing values become ''."""
    return "" if pd.isna(value) else str(value)


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
        unmaskable = 0
        unparsed = 0
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size]
            prompts: list[str] = []
            metas: list[tuple[str, str, str, str]] = []
            for _, row in batch.iterrows():
                query_id = _cell(row["id"])
                query = _cell(row["query"])
                ad_id = _cell(row["ad_id"])
                position = _cell(row["position"])
                try:
                    masked, ad_text = mask_ad_in_response(
                        _cell(row["response_with_ad"]), _cell(row["headline"]), _cell(row["description"]), _cell(row["cta"])
                    )
                except AdBlockNotFoundError:
                    print(f"  skip id={query_id!r} position={position!r}: ad block not found, cannot mask")
                    unmaskable += 1
                    continue
                prompts.append(build_user_prompt(query, masked, ad_text))
                metas.append((query_id, query, ad_id, position))
            raw_texts = self._judge.generate(prompts) if prompts else []
            for meta, raw in zip(metas, raw_texts):
                query_id, query, ad_id, position = meta
                try:
                    score, reason = parse_placement_score(raw)
                except (ValueError, TypeError, KeyError) as exc:
                    print(f"  parse fail id={query_id!r} position={position!r}: {exc}")
                    unparsed += 1
                    continue
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
        if buffer:
            self._writer.save(buffer, output_path, append=wrote_any)
            all_rows.extend(buffer)
        if unmaskable:
            print(f"Skipped {unmaskable:,} placements whose ad block could not be masked (not saved; retried on resume).")
        _report_unparsed(unparsed)
        return all_rows


@dataclass(frozen=True)
class _PreferencePair:
    query_id: str
    query: str
    ad_1: Ad
    ad_2: Ad
    is_swapped: bool


class EvaluateAdPreferenceService:
    """Judges the query's assigned ad against candidate ads, in both orders, to measure preference.

    `same_category` compares against the other ads in the assigned ad's category;
    `cross_category` compares against every ad in all other categories.
    """
    def __init__(
        self,
        writer: PreferenceScoreCsvRepository,
        judge: VllmPreferenceJudge,
        candidates: PreferenceCandidates = "same_category",
    ) -> None:
        self._writer = writer
        self._judge = judge
        self._candidates = candidates

    def _candidate_ads(self, selected: Ad, ads_by_category: dict[str, list[Ad]]) -> Iterable[Ad]:
        if self._candidates == "same_category":
            return (ad for ad in ads_by_category[selected.category] if ad.id != selected.id)
        return (
            ad
            for category, ads in ads_by_category.items()
            if category != selected.category
            for ad in ads
        )

    def run(
        self,
        assignments_path: Path,
        ads_path: Path,
        output_path: Path,
        limit: int | None = None,
        batch_size: int = 100,
    ) -> list[PairwisePreferenceScore]:
        assignments = pd.read_csv(assignments_path)
        if limit is not None:
            assignments = assignments.head(limit).copy()

        all_ads = JsonlAdRepository(ads_path).load()
        ads_by_id = {ad.id: ad for ad in all_ads}
        ads_by_category: dict[str, list[Ad]] = defaultdict(list)
        for ad in all_ads:
            if ad.category:
                ads_by_category[ad.category].append(ad)

        done = load_done_keys(output_path, ["query_id", "ad_1_id", "ad_2_id", "is_swapped"])
        wrote_any = done is not None
        scored_keys = done or set()
        if wrote_any:
            print(f"Resuming: {len(scored_keys):,} pairs already scored.")

        pairs: list[_PreferencePair] = []
        for _, row in assignments.iterrows():
            query_id = str(row["id"])
            query = str(row["query"])
            selected = ads_by_id.get(str(row["ad_id"]))
            if not selected or not selected.category:
                continue
            for other in self._candidate_ads(selected, ads_by_category):
                if (query_id, selected.id, other.id, "False") not in scored_keys:
                    pairs.append(_PreferencePair(query_id, query, selected, other, is_swapped=False))
                if (query_id, other.id, selected.id, "True") not in scored_keys:
                    pairs.append(_PreferencePair(query_id, query, other, selected, is_swapped=True))

        label = "preference" if self._candidates == "same_category" else "cross-category preference"
        print(f"Total {label} pairs to score: {len(pairs):,}")

        all_rows: list[PairwisePreferenceScore] = []
        buffer: list[PairwisePreferenceScore] = []
        unparsed = 0
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            prompts = [
                build_pairwise_preference_prompt(pair.query, pair.ad_1.embed_text, pair.ad_2.embed_text)
                for pair in batch
            ]
            raw_texts = self._judge.generate(prompts)
            for pair, raw in zip(batch, raw_texts):
                try:
                    winner_tag, confidence = parse_pairwise_preference(raw)
                    winner_ad_id = pair.ad_1.id if winner_tag == "ad_1" else pair.ad_2.id
                except Exception as exc:
                    print(f"  parse fail query_id={pair.query_id!r}: {exc}")
                    unparsed += 1
                    continue
                buffer.append(
                    PairwisePreferenceScore(
                        query_id=pair.query_id,
                        query=pair.query,
                        ad_1_id=pair.ad_1.id,
                        ad_2_id=pair.ad_2.id,
                        winner_ad_id=winner_ad_id,
                        confidence=confidence,
                        is_swapped=pair.is_swapped,
                    )
                )

            finished = min(start + batch_size, len(pairs))
            print(f"  scored {finished:,}/{len(pairs):,}")
            if len(buffer) >= CHECKPOINT_EVERY or finished == len(pairs):
                self._writer.save(buffer, output_path, append=wrote_any)
                wrote_any = True
                all_rows.extend(buffer)
                buffer = []

        _report_unparsed(unparsed)
        return all_rows


class EvaluatePlacementComparisonService:
    """Coordinates batch generation of pairwise placement comparisons."""
    def __init__(
        self,
        writer: PlacementComparisonScoreCsvRepository,
        judge: VllmPlacementComparisonJudge,
    ) -> None:
        self._writer = writer
        self._judge = judge

    def run(
        self,
        positions_path: Path,
        ads_path: Path,
        output_path: Path,
        limit: int | None = None,
        batch_size: int = 100,
    ) -> list[PlacementComparisonScore]:
        positions = pd.read_csv(positions_path)
        ads = pd.read_csv(ads_path)
        # Ad headline/description/cta are needed to rebuild and mask the ad block.
        merged = positions.merge(ads, on=["id", "ad_id"], suffixes=("", "_ad"))
        grouped = merged.groupby(["id", "ad_id"])

        done = load_done_keys(output_path, ["query_id", "ad_id", "pos_1", "pos_2", "is_swapped"])
        wrote_any = done is not None
        scored_keys = done or set()
        if wrote_any:
            print(f"Resuming: {len(scored_keys):,} pairs already scored.")

        pairs_to_test = []
        unmaskable = 0
        for (query_id, ad_id), group in grouped:
            query_id = str(query_id)
            ad_id = str(ad_id)
            query = str(group["query"].iloc[0])
            headline = _cell(group["headline"].iloc[0])
            description = _cell(group["description"].iloc[0])
            cta = _cell(group["cta"].iloc[0])
            responses = {row["position"]: str(row["response_with_ad"]) for _, row in group.iterrows()}
            if "semantic" not in responses:
                continue

            for other_pos in ["first", "middle", "last"]:
                if other_pos not in responses:
                    continue
                try:
                    masked_semantic, ad_block = mask_ad_in_response(responses["semantic"], headline, description, cta)
                    masked_other, _ = mask_ad_in_response(responses[other_pos], headline, description, cta)
                except AdBlockNotFoundError:
                    print(f"  skip query_id={query_id!r} semantic vs {other_pos}: ad block not found, cannot mask")
                    unmaskable += 1
                    continue
                orders = (
                    ("semantic", other_pos, masked_semantic, masked_other, False),
                    (other_pos, "semantic", masked_other, masked_semantic, True),
                )
                for pos_1, pos_2, resp_1, resp_2, is_swapped in orders:
                    if (query_id, ad_id, pos_1, pos_2, str(is_swapped)) in scored_keys:
                        continue
                    pairs_to_test.append({
                        "query_id": query_id,
                        "query": query,
                        "ad_id": ad_id,
                        "ad_block": ad_block,
                        "pos_1": pos_1,
                        "pos_2": pos_2,
                        "resp_1": resp_1,
                        "resp_2": resp_2,
                        "is_swapped": is_swapped,
                    })

        if limit is not None:
            pairs_to_test = pairs_to_test[:limit]

        if unmaskable:
            print(f"Skipped {unmaskable:,} position comparisons whose ad block could not be masked (retried on resume).")
        print(f"Total placement pairs to score: {len(pairs_to_test):,}")

        all_rows: list[PlacementComparisonScore] = []
        buffer: list[PlacementComparisonScore] = []
        unparsed = 0
        for start in range(0, len(pairs_to_test), batch_size):
            batch = pairs_to_test[start:start + batch_size]
            prompts = [
                build_placement_comparison_prompt(item["query"], item["ad_block"], item["resp_1"], item["resp_2"])
                for item in batch
            ]
            raw_texts = self._judge.generate(prompts)
            for item, raw in zip(batch, raw_texts):
                try:
                    winner_tag, confidence = parse_placement_comparison(raw)
                    winner_pos = item["pos_1"] if winner_tag == "pos_1" else item["pos_2"]
                except Exception as exc:
                    print(f"  parse fail query_id={item['query_id']!r}: {exc}")
                    unparsed += 1
                    continue
                buffer.append(
                    PlacementComparisonScore(
                        query_id=item["query_id"],
                        query=item["query"],
                        ad_id=item["ad_id"],
                        pos_1=item["pos_1"],
                        pos_2=item["pos_2"],
                        winner_pos=winner_pos,
                        confidence=confidence,
                        is_swapped=item["is_swapped"],
                    )
                )

            finished = min(start + batch_size, len(pairs_to_test))
            print(f"  scored {finished:,}/{len(pairs_to_test):,}")
            if len(buffer) >= CHECKPOINT_EVERY or finished == len(pairs_to_test):
                self._writer.save(buffer, output_path, append=wrote_any)
                wrote_any = True
                all_rows.extend(buffer)
                buffer = []

        _report_unparsed(unparsed)
        return all_rows
