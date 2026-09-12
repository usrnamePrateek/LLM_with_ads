"""CLI script to run the LLM pairwise judge on placement comparisons."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.ad_evaluation.config import (
    JUDGE_BATCH_SIZE,
    JUDGE_GPU_MEMORY_UTILIZATION,
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MODEL_NAME,
)
from src.ad_evaluation.llm.preference_judge import VllmPreferenceJudge
from src.ad_evaluation.repository import PlacementComparisonScoreCsvRepository
from src.ad_evaluation.services import EvaluatePlacementComparisonService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score placement comparisons pairwise (semantic vs first/middle/last) with Llama 3.3 70B (vLLM)."
    )
    parser.add_argument(
        "--positions", 
        type=Path, 
        default=Path("data/processed/generation/bulk_ads/query_ad_positions.csv"),
        help="Path to the generated responses with ad placements."
    )
    parser.add_argument(
        "--ads", 
        type=Path, 
        default=Path("data/processed/generation/bulk_ads/query_top1_ads.csv"),
        help="Path to the assignments to get ad headline/description/cta."
    )
    parser.add_argument(
        "--output", 
        type=Path, 
        default=Path("data/evaluation/placement_comparison_scores.csv"),
        help="Path to save the comparison scores."
    )
    parser.add_argument("--model", default=JUDGE_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=JUDGE_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=JUDGE_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--batch-size", type=int, default=JUDGE_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    
    args = parser.parse_args()

    judge = VllmPreferenceJudge(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    
    writer = PlacementComparisonScoreCsvRepository()
    service = EvaluatePlacementComparisonService(
        writer=writer,
        judge=judge,
    )
    
    try:
        service.run(
            positions_path=args.positions,
            ads_path=args.ads,
            output_path=args.output,
            limit=args.limit,
            batch_size=args.batch_size,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user. Checkpoint has been saved.")
        return 130
    except Exception as exc:
        print(f"\nEvaluation failed: {exc}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
