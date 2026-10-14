"""CLI script to benchmark and compare ad retrieval latency against LLM response generation latency."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.speed_comparison.config import (
    CHECKPOINT_EVERY,
    DEFAULT_BATCH_SIZE,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_QUERIES,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_QUERIES_PARQUET,
    DEFAULT_RETRIEVAL_DEVICE,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
)
from src.speed_comparison.services import SpeedComparisonOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark and compare the speed of LLM response generation with ad retrieval."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PARQUET,
        help="Path to queries parquet or CSV dataset.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Path to the FAISS index and ads metadata directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Path to output comparison CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to output aggregated summary JSON.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_NUM_QUERIES,
        help="Number of queries to benchmark (default: 1000). Set to -1 or 0 for unlimited.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for benchmarking (default: 1 for single-stream interactive latency).",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"vLLM model identifier (default: {DEFAULT_LLM_MODEL}).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Maximum model context length for vLLM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Maximum tokens to generate per query.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        help="Fraction of GPU VRAM allocated to vLLM (default: 0.90).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for LLM generation.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top ads to retrieve per query (default: 1).",
    )
    parser.add_argument(
        "--retrieval-device",
        type=str,
        choices=["cuda", "cpu"],
        default=DEFAULT_RETRIEVAL_DEVICE,
        help=f"Device for BGE-M3 embedding encoder (default: {DEFAULT_RETRIEVAL_DEVICE}).",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Benchmark only the ad retrieval phase (embedding + vector search) without loading or running the LLM.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        help="Flush incremental results every N queries (default: 50).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = None if args.limit in (None, 0, -1) else args.limit

    orchestrator = SpeedComparisonOrchestrator()
    orchestrator.run(
        queries_path=args.queries_path,
        index_dir=args.index_dir,
        output_csv=args.output,
        summary_json=args.summary_output,
        limit=limit,
        batch_size=args.batch_size,
        llm_model=args.llm_model,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature,
        top_k=args.top_k,
        retrieval_device=args.retrieval_device,
        retrieval_only=args.retrieval_only,
        checkpoint_every=args.checkpoint_every,
    )


if __name__ == "__main__":
    main()
