"""CLI script to run the LLM pairwise preference judge on ad assignments."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_evaluation.config import (
    DEFAULT_ADS_CSV,
    JUDGE_BATCH_SIZE,
    JUDGE_GPU_MEMORY_UTILIZATION,
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MODEL_NAME,
)
from src.ad_evaluation.llm.preference_judge import VllmPreferenceJudge
from src.ad_evaluation.repository import PreferenceScoreCsvRepository
from src.ad_evaluation.services import EvaluateAdPreferenceService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ad preferences pairwise with Llama 3.3 70B (vLLM)."
    )
    # The default assignments path
    parser.add_argument("--assignments", type=Path, default=Path("data/processed/generation/bulk_ads/query_top1_ads.csv"))
    parser.add_argument("--ads", type=Path, default=Path("data/processed/ads/ads.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation/preference_scores.csv"))
    parser.add_argument("--model", default=JUDGE_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=JUDGE_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=JUDGE_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--batch-size", type=int, default=JUDGE_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge = VllmPreferenceJudge(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    EvaluateAdPreferenceService(
        writer=PreferenceScoreCsvRepository(),
        judge=judge,
    ).run(
        assignments_path=args.assignments,
        ads_path=args.ads,
        output_path=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
