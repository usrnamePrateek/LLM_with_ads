"""Script entry point for categorizing LMSYS Chatbot Arena queries using an LLM."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_generation.config import (
    CATEGORY_BATCH_SIZE,
    CATEGORY_COLUMNS,
    CATEGORY_GPU_MEMORY_UTILIZATION,
    CATEGORY_MAX_MODEL_LEN,
    CATEGORY_MODEL_NAME,
    DEFAULT_CATEGORIES_OUTPUT,
    DEFAULT_QUERIES_INPUT,
    QUERY_COLUMN,
)
from src.ad_generation.llm.category_generator import VllmCategoryGenerator
from src.ad_generation.repository import QueryFrameRepository
from src.ad_generation.services import CategorizeQueriesService, row_already_categorized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Categorize LMArena queries with local Qwen3-32B fp16 via vLLM."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_QUERIES_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_CATEGORIES_OUTPUT)
    parser.add_argument("--query-column", default=QUERY_COLUMN)
    parser.add_argument("--model", default=CATEGORY_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=CATEGORY_BATCH_SIZE)
    parser.add_argument("--max-model-len", type=int, default=CATEGORY_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=CATEGORY_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    generator = VllmCategoryGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(
        "vLLM ready.",
        "model:", args.model,
        "dtype: fp16",
        "batch_size:", args.batch_size,
    )

    repository = QueryFrameRepository()
    source_df = repository.load(args.input, args.query_column)
    frame = repository.merge_checkpoint(args.output, source_df)
    already = frame.apply(row_already_categorized, axis=1).sum()
    print(f"Resumed {int(already):,} categorized rows; {len(frame) - int(already):,} remaining")

    categorized = CategorizeQueriesService(generator=generator, repository=repository).run(
        frame,
        output_path=args.output,
        query_column=args.query_column,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(categorized[CATEGORY_COLUMNS].head())


if __name__ == "__main__":
    main()
