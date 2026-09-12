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
        if buffer:
            self._writer.save(buffer, output_path, append=wrote_any)
            all_rows.extend(buffer)
        return all_rows


class EvaluateAdPreferenceService:
    """Coordinates batch generation of pairwise preference comparisons."""
    def __init__(
        self,
        writer: 'src.ad_evaluation.repository.PreferenceScoreCsvRepository',
        judge: 'src.ad_evaluation.llm.preference_judge.VllmPreferenceJudge',
    ) -> None:
        self._writer = writer
        self._judge = judge

    def run(
        self,
        assignments_path: Path,
        ads_path: Path,
        output_path: Path,
        limit: int | None = None,
        batch_size: int = 100,
    ) -> list['src.ad_evaluation.entities.PairwisePreferenceScore']:
        from src.ad_indexing.repository import JsonlAdRepository
        from src.ad_evaluation.entities import PairwisePreferenceScore
        from src.ad_evaluation.llm.prompts import build_pairwise_preference_prompt
        from src.ad_evaluation.llm.parsing import parse_pairwise_preference
        from collections import defaultdict

        # Load assignments
        assignments = pd.read_csv(assignments_path)
        if limit is not None:
            assignments = assignments.head(limit).copy()

        # Load ads to find category matches
        repo = JsonlAdRepository(ads_path)
        all_ads = repo.load()
        
        ads_by_id = {ad.id: ad for ad in all_ads}
        ads_by_category = defaultdict(list)
        for ad in all_ads:
            if ad.category:
                ads_by_category[ad.category].append(ad)

        wrote_any = False
        scored_keys = set()
        if output_path.exists():
            existing = pd.read_csv(output_path, dtype=str)
            if "query_id" in existing.columns and "ad_1_id" in existing.columns and "ad_2_id" in existing.columns and "is_swapped" in existing.columns:
                scored_keys = set(
                    zip(existing["query_id"], existing["ad_1_id"], existing["ad_2_id"], existing["is_swapped"])
                )
                wrote_any = True
                print(f"Resuming: {len(scored_keys):,} pairs already scored.")

        # Build testing pairs
        pairs_to_test = []
        for _, row in assignments.iterrows():
            query_id = str(row["id"])
            query = str(row["query"])
            selected_ad_id = str(row["ad_id"])
            
            selected_ad = ads_by_id.get(selected_ad_id)
            if not selected_ad or not selected_ad.category:
                continue
                
            same_category_ads = ads_by_category[selected_ad.category]
            for other_ad in same_category_ads:
                if other_ad.id == selected_ad_id:
                    continue
                
                # Forward pair
                fwd_key = (query_id, selected_ad_id, other_ad.id, "False")
                if fwd_key not in scored_keys:
                    pairs_to_test.append({
                        "query_id": query_id,
                        "query": query,
                        "ad_1": selected_ad,
                        "ad_2": other_ad,
                        "is_swapped": False
                    })
                    
                # Swapped pair
                rev_key = (query_id, other_ad.id, selected_ad_id, "True")
                if rev_key not in scored_keys:
                    pairs_to_test.append({
                        "query_id": query_id,
                        "query": query,
                        "ad_1": other_ad,
                        "ad_2": selected_ad,
                        "is_swapped": True
                    })

        print(f"Total preference pairs to score: {len(pairs_to_test):,}")

        all_rows = []
        buffer = []
        for start in range(0, len(pairs_to_test), batch_size):
            batch = pairs_to_test[start:start + batch_size]
            prompts = []
            for item in batch:
                ad_1, ad_2 = item["ad_1"], item["ad_2"]
                prompts.append(build_pairwise_preference_prompt(item["query"], ad_1.embed_text, ad_2.embed_text))
                
            raw_texts = self._judge.generate(prompts)
            for item, raw in zip(batch, raw_texts):
                try:
                    winner_tag, confidence = parse_pairwise_preference(raw)
                    winner_ad_id = item["ad_1"].id if winner_tag == "ad_1" else item["ad_2"].id
                except Exception as exc:
                    print(f"  parse fail query_id={item['query_id']!r}: {exc}")
                    winner_ad_id = "ERROR"
                    confidence = f"parse_error: {exc}"
                    
                score = PairwisePreferenceScore(
                    query_id=item["query_id"],
                    query=item["query"],
                    ad_1_id=item["ad_1"].id,
                    ad_2_id=item["ad_2"].id,
                    winner_ad_id=winner_ad_id,
                    confidence=confidence,
                    is_swapped=item["is_swapped"],
                )
                buffer.append(score)
                
            done = min(start + batch_size, len(pairs_to_test))
            print(f"  scored {done:,}/{len(pairs_to_test):,}")
            if len(buffer) >= 1000 or done == len(pairs_to_test):
                self._writer.save(buffer, output_path, append=wrote_any)
                wrote_any = True
                all_rows.extend(buffer)
                buffer = []
                
        return all_rows

class EvaluateCrossCategoryAdPreferenceService:
    """Coordinates exhaustive batch generation of pairwise preference comparisons against ads from OTHER categories."""
    def __init__(
        self,
        writer: 'src.ad_evaluation.repository.PreferenceScoreCsvRepository',
        judge: 'src.ad_evaluation.llm.preference_judge.VllmPreferenceJudge',
    ) -> None:
        self._writer = writer
        self._judge = judge

    def run(
        self,
        assignments_path: Path,
        ads_path: Path,
        output_path: Path,
        limit: int | None = None,
        batch_size: int = 100,
    ) -> list['src.ad_evaluation.entities.PairwisePreferenceScore']:
        from src.ad_indexing.repository import JsonlAdRepository
        from src.ad_evaluation.entities import PairwisePreferenceScore
        from src.ad_evaluation.llm.prompts import build_pairwise_preference_prompt
        from src.ad_evaluation.llm.parsing import parse_pairwise_preference
        from collections import defaultdict

        # Load assignments
        assignments = pd.read_csv(assignments_path)
        if limit is not None:
            assignments = assignments.head(limit).copy()

        # Load ads to find category matches
        repo = JsonlAdRepository(ads_path)
        all_ads = repo.load()
        
        ads_by_id = {ad.id: ad for ad in all_ads}
        ads_by_category = defaultdict(list)
        for ad in all_ads:
            if ad.category:
                ads_by_category[ad.category].append(ad)

        wrote_any = False
        scored_keys = set()
        if output_path.exists():
            existing = pd.read_csv(output_path, dtype=str)
            if "query_id" in existing.columns and "ad_1_id" in existing.columns and "ad_2_id" in existing.columns and "is_swapped" in existing.columns:
                scored_keys = set(
                    zip(existing["query_id"], existing["ad_1_id"], existing["ad_2_id"], existing["is_swapped"])
                )
                wrote_any = True
                print(f"Resuming: {len(scored_keys):,} pairs already scored.")

        # Build testing pairs
        pairs_to_test = []
        for _, row in assignments.iterrows():
            query_id = str(row["id"])
            query = str(row["query"])
            selected_ad_id = str(row["ad_id"])
            
            selected_ad = ads_by_id.get(selected_ad_id)
            if not selected_ad or not selected_ad.category:
                continue
                
            for category, ads in ads_by_category.items():
                if category == selected_ad.category:
                    continue # SKIP SAME CATEGORY
                    
                for other_ad in ads:
                    # Forward pair
                    fwd_key = (query_id, selected_ad_id, other_ad.id, "False")
                    if fwd_key not in scored_keys:
                        pairs_to_test.append({
                            "query_id": query_id,
                            "query": query,
                            "ad_1": selected_ad,
                            "ad_2": other_ad,
                            "is_swapped": False
                        })
                        
                    # Swapped pair
                    rev_key = (query_id, other_ad.id, selected_ad_id, "True")
                    if rev_key not in scored_keys:
                        pairs_to_test.append({
                            "query_id": query_id,
                            "query": query,
                            "ad_1": other_ad,
                            "ad_2": selected_ad,
                            "is_swapped": True
                        })

        print(f"Total cross-category preference pairs to score: {len(pairs_to_test):,}")

        all_rows = []
        buffer = []
        for start in range(0, len(pairs_to_test), batch_size):
            batch = pairs_to_test[start:start + batch_size]
            prompts = []
            for item in batch:
                ad_1, ad_2 = item["ad_1"], item["ad_2"]
                prompts.append(build_pairwise_preference_prompt(item["query"], ad_1.embed_text, ad_2.embed_text))
                
            raw_texts = self._judge.generate(prompts)
            for item, raw in zip(batch, raw_texts):
                try:
                    winner_tag, confidence = parse_pairwise_preference(raw)
                    winner_ad_id = item["ad_1"].id if winner_tag == "ad_1" else item["ad_2"].id
                except Exception as exc:
                    print(f"  parse fail query_id={item['query_id']!r}: {exc}")
                    winner_ad_id = "ERROR"
                    confidence = f"parse_error: {exc}"
                    
                score = PairwisePreferenceScore(
                    query_id=item["query_id"],
                    query=item["query"],
                    ad_1_id=item["ad_1"].id,
                    ad_2_id=item["ad_2"].id,
                    winner_ad_id=winner_ad_id,
                    confidence=confidence,
                    is_swapped=item["is_swapped"],
                )
                buffer.append(score)
                
            done = min(start + batch_size, len(pairs_to_test))
            print(f"  scored {done:,}/{len(pairs_to_test):,}")
            if len(buffer) >= 1000 or done == len(pairs_to_test):
                self._writer.save(buffer, output_path, append=wrote_any)
                wrote_any = True
                all_rows.extend(buffer)
                buffer = []
                
        return all_rows

class EvaluatePlacementComparisonService:
    """Coordinates batch generation of pairwise placement comparisons."""
    def __init__(
        self,
        writer: 'src.ad_evaluation.repository.PlacementComparisonScoreCsvRepository',
        judge: 'src.ad_evaluation.llm.preference_judge.VllmPreferenceJudge',
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
    ) -> list['src.ad_evaluation.entities.PlacementComparisonScore']:
        from src.ad_evaluation.entities import PlacementComparisonScore
        from src.ad_evaluation.llm.prompts import build_placement_comparison_prompt
        from src.ad_evaluation.llm.parsing import parse_placement_comparison
        from src.ad_integration.core.placement import format_ad_block
        
        # Load positions and ads
        positions = pd.read_csv(positions_path)
        ads = pd.read_csv(ads_path)
        
        # We need headline, description, cta for the ad block
        # Merge to get ad info
        merged = positions.merge(ads, left_on="ad_id", right_on="id", suffixes=("", "_ad"))
        
        # Group by query_id and ad_id
        grouped = merged.groupby(["id", "ad_id"])
        
        wrote_any = False
        scored_keys = set()
        if output_path.exists():
            existing = pd.read_csv(output_path, dtype=str)
            if "query_id" in existing.columns and "ad_id" in existing.columns and "pos_1" in existing.columns and "pos_2" in existing.columns and "is_swapped" in existing.columns:
                scored_keys = set(
                    zip(existing["query_id"], existing["ad_id"], existing["pos_1"], existing["pos_2"], existing["is_swapped"])
                )
                wrote_any = True
                print(f"Resuming: {len(scored_keys):,} pairs already scored.")

        pairs_to_test = []
        for (query_id, ad_id), group in grouped:
            query_id = str(query_id)
            ad_id = str(ad_id)
            query = str(group["query"].iloc[0])
            
            # Extract ad block
            headline = str(group["headline"].iloc[0])
            description = str(group["description"].iloc[0])
            cta = str(group["cta"].iloc[0])
            ad_block = format_ad_block(headline, description, cta)
            
            # Map positions to responses
            responses = {row["position"]: str(row["response_with_ad"]) for _, row in group.iterrows()}
            
            if "semantic" not in responses:
                continue
                
            semantic_response = responses["semantic"]
            
            for other_pos in ["first", "middle", "last"]:
                if other_pos not in responses:
                    continue
                other_response = responses[other_pos]
                
                # Forward pair (semantic vs other)
                fwd_key = (query_id, ad_id, "semantic", other_pos, "False")
                if fwd_key not in scored_keys:
                    pairs_to_test.append({
                        "query_id": query_id,
                        "query": query,
                        "ad_id": ad_id,
                        "ad_block": ad_block,
                        "pos_1": "semantic",
                        "pos_2": other_pos,
                        "resp_1": semantic_response,
                        "resp_2": other_response,
                        "is_swapped": False
                    })
                    
                # Swapped pair (other vs semantic)
                rev_key = (query_id, ad_id, other_pos, "semantic", "True")
                if rev_key not in scored_keys:
                    pairs_to_test.append({
                        "query_id": query_id,
                        "query": query,
                        "ad_id": ad_id,
                        "ad_block": ad_block,
                        "pos_1": other_pos,
                        "pos_2": "semantic",
                        "resp_1": other_response,
                        "resp_2": semantic_response,
                        "is_swapped": True
                    })

        if limit is not None:
            pairs_to_test = pairs_to_test[:limit]

        print(f"Total placement pairs to score: {len(pairs_to_test):,}")

        all_rows = []
        buffer = []
        for start in range(0, len(pairs_to_test), batch_size):
            batch = pairs_to_test[start:start + batch_size]
            prompts = []
            for item in batch:
                prompts.append(build_placement_comparison_prompt(
                    item["query"], 
                    item["ad_block"], 
                    item["resp_1"], 
                    item["resp_2"]
                ))
                
            raw_texts = self._judge.generate(prompts)
            for item, raw in zip(batch, raw_texts):
                try:
                    winner_tag, confidence = parse_placement_comparison(raw)
                    winner_pos = item["pos_1"] if winner_tag == "pos_1" else item["pos_2"]
                except Exception as exc:
                    print(f"  parse fail query_id={item['query_id']!r}: {exc}")
                    winner_pos = "ERROR"
                    confidence = f"parse_error: {exc}"
                    
                score = PlacementComparisonScore(
                    query_id=item["query_id"],
                    query=item["query"],
                    ad_id=item["ad_id"],
                    pos_1=item["pos_1"],
                    pos_2=item["pos_2"],
                    winner_pos=winner_pos,
                    confidence=confidence,
                    is_swapped=item["is_swapped"],
                )
                buffer.append(score)
                
            done = min(start + batch_size, len(pairs_to_test))
            print(f"  scored {done:,}/{len(pairs_to_test):,}")
            if len(buffer) >= 1000 or done == len(pairs_to_test):
                self._writer.save(buffer, output_path, append=wrote_any)
                wrote_any = True
                all_rows.extend(buffer)
                buffer = []
                
        return all_rows
