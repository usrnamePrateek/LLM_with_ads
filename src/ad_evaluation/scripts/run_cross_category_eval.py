import argparse
import sys
from pathlib import Path

from src.ad_evaluation.config import (
    DEFAULT_ADS_CSV,
    DEFAULT_CROSS_CATEGORY_PREF_CSV,
)
from src.ad_evaluation.llm.preference_judge import VllmPreferenceJudge
from src.ad_evaluation.repository import PreferenceScoreCsvRepository
from src.ad_evaluation.services import EvaluateAdPreferenceService
from src.common.shared_config import DEFAULT_ADS_JSONL

def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pairwise cross-category ad preferences.")
    parser.add_argument(
        "--assignments",
        type=Path,
        default=DEFAULT_ADS_CSV,
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
        default=DEFAULT_CROSS_CATEGORY_PREF_CSV,
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
        default=128,
        help="vLLM batch size",
    )
    parser.add_argument(
        "--single-order",
        action="store_true",
        help="Judge each pair once (assigned ad as ad_1) instead of in both orders",
    )
    args = parser.parse_args()

    judge = VllmPreferenceJudge()
    writer = PreferenceScoreCsvRepository()
    service = EvaluateAdPreferenceService(writer=writer, judge=judge, candidates="cross_category")

    try:
        service.run(
            assignments_path=args.assignments,
            ads_path=args.ads,
            output_path=args.output,
            limit=args.limit,
            batch_size=args.batch_size,
            both_orders=not args.single_order,
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
