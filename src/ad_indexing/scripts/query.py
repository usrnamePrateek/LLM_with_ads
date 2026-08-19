"""CLI script to test the vector index by performing a semantic search."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.ad_indexing.config import DEFAULT_INDEX_DIR, DEFAULT_TOP_K
from src.ad_indexing.embeddings.bge_m3 import BgeM3Encoder
from src.ad_indexing.services import SearchAdsService
from src.ad_indexing.store.faiss_store import FaissVectorStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query the FAISS ad index.")
    parser.add_argument("query", help="User query text")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = FaissVectorStore(args.index_dir)
    store.load()
    encoder = BgeM3Encoder()
    hits = SearchAdsService(encoder=encoder, store=store).search(args.query, args.k)
    if not hits:
        print("No ads found.")
        return
    for rank, hit in enumerate(hits, start=1):
        ad = hit.ad
        print(f"[{rank}] score={hit.score:.4f} | {ad.domain} | ad_id={ad.ad_id}")
        print(f"    headline: {ad.headline}")
        print(f"    description: {ad.description}")
        if ad.cta:
            print(f"    cta: {ad.cta}")
        print()


if __name__ == "__main__":
    main()
