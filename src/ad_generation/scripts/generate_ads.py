from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_generation.config import (
    AD_GPU_MEMORY_UTILIZATION,
    AD_MAX_MODEL_LEN,
    AD_MODEL_NAME,
    DEFAULT_ADS_JSONL,
    DEFAULT_DOMAINS_TXT,
)
from src.ad_generation.llm.ad_generator import VllmAdGenerator
from src.ad_generation.repository import AdJsonlRepository, TextDomainRepository
from src.ad_generation.services import GenerateAdsService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate two ads per domain with vLLM structured output.")
    parser.add_argument("--input", type=Path, default=DEFAULT_DOMAINS_TXT)
    parser.add_argument("--output", type=Path, default=DEFAULT_ADS_JSONL)
    parser.add_argument("--model", default=AD_MODEL_NAME)
    parser.add_argument("--max-model-len", type=int, default=AD_MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=AD_GPU_MEMORY_UTILIZATION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generator = VllmAdGenerator(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    GenerateAdsService(
        domains=TextDomainRepository(args.input),
        generator=generator,
        writer=AdJsonlRepository(args.output),
    ).run()


if __name__ == "__main__":
    main()
