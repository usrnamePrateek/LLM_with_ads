"""CLI script to run the LLM judge on synthetically placed ads."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_evaluation.config import (
    DEFAULT_ADS_CSV,
    DEFAULT_POSITIONS_CSV,
    DEFAULT_SCORES_CSV,
    JUDGE_BATCH_SIZE,
    JUDGE_GPU_MEMORY_UTILIZATION,
    JUDGE_MAX_MODEL_LEN,
    JUDGE_MODEL_NAME,
)
from src.ad_evaluation.llm.judge import VllmPlacementJudge
from src.ad_evaluation.repository import PlacementInputRepository, PlacementScoreCsvRepository
from src.ad_evaluation.services import ScorePlacementsService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ad placement quality with Llama 3.3 70B (vLLM)."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_POSITIONS_CSV)
    parser.add_argument("--ads", type=Path, default=DEFAULT_ADS_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORES_CSV)
    parser.add_argument("--model", default=JUDGE_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=JUDGE_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=JUDGE_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--batch-size", type=int, default=JUDGE_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    judge = VllmPlacementJudge(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    ScorePlacementsService(
        inputs=PlacementInputRepository(),
        writer=PlacementScoreCsvRepository(),
        judge=judge,
    ).run(
        positions_path=args.input,
        ads_path=args.ads,
        output_path=args.output,
        limit=args.limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
