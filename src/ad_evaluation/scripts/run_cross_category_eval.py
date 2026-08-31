import argparse
import sys
from pathlib import Path

from src.ad_evaluation.config import (
    DEFAULT_ASSIGNMENTS_CSV,
    DEFAULT_ADS_JSONL,
)
from src.ad_evaluation.llm.preference_judge import VllmPreferenceJudge
from src.ad_evaluation.repository import PreferenceScoreCsvRepository
from src.ad_evaluation.services import EvaluateCrossCategoryAdPreferenceService

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pairwise cross-category ad preferences.")
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ASSIGNMENTS_CSV,
        help="Path to query_top1_ads.csv",
    )
    parser.add_argument(
        "--ads",
        type=Path,
        default=DEFAULT_ADS_JSONL,
        help="Path to ads.jsonl containing all generated ad creatives",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/cross_category_preference_scores.csv"),
        help="Path to save preference scores",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of query assignments to process (for testing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="vLLM batch size",
    )
    args = parser.parse_args()

    judge = VllmPreferenceJudge()
    writer = PreferenceScoreCsvRepository()
    service = EvaluateCrossCategoryAdPreferenceService(writer=writer, judge=judge)

    try:
        service.run(
            assignments_path=args.assignments,
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
