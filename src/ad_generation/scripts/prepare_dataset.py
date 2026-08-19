from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_generation.config import DEFAULT_ARENA_OUTPUT_DIR
from src.ad_generation.services import PrepareArenaDatasetService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Arena Human Preference data for downstream use."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARENA_OUTPUT_DIR,
        help="Directory for parquet + HF disk save",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    PrepareArenaDatasetService().run(args.output_dir)


if __name__ == "__main__":
    main()
