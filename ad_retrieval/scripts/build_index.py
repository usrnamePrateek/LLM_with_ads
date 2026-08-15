from __future__ import annotations

import argparse
from pathlib import Path

from ad_retrieval.config import DEFAULT_ADS_JSONL, DEFAULT_INDEX_DIR
from ad_retrieval.embeddings.bge_m3 import BgeM3Encoder
from ad_retrieval.repository import JsonlAdRepository
from ad_retrieval.services import IndexAdsService
from ad_retrieval.store.faiss_store import FaissVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index of ads with BGE-M3 embeddings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_ADS_JSONL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading ads from {args.input}")
    encoder = BgeM3Encoder()
    store = FaissVectorStore(args.index_dir)
    count = IndexAdsService(
        repository=JsonlAdRepository(args.input),
        encoder=encoder,
        store=store,
    ).run()
    print(f"Indexed {count} ads into {args.index_dir}")


if __name__ == "__main__":
    main()
