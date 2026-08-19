from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_indexing.config import DEFAULT_ADS_JSONL, DEFAULT_INDEX_DIR
from src.ad_indexing.embeddings.bge_m3 import BgeM3Encoder
from src.ad_indexing.repository import JsonlAdRepository
from src.ad_indexing.services import IndexAdsService
from src.ad_indexing.store.faiss_store import FaissVectorStore


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
