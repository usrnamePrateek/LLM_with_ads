from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from ad_retrieval.config import ADS_SIDECAR_FILENAME, INDEX_FILENAME
from ad_retrieval.models import Ad, ScoredAd


class FaissVectorStore:
    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._index_path = index_dir / INDEX_FILENAME
        self._sidecar_path = index_dir / ADS_SIDECAR_FILENAME
        self._index: faiss.Index | None = None
        self._ads: list[Ad] = []
        self._id_to_row: dict[str, int] = {}

    def _rebuild_id_map(self) -> None:
        self._id_to_row = {ad.id: i for i, ad in enumerate(self._ads)}

    def upsert(self, ids: list[str], vectors: np.ndarray, ads: list[Ad]) -> None:
        if len(ids) != len(ads) or len(ids) != len(vectors):
            raise ValueError("ids, vectors, and ads must have the same length")
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D array")

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)
        self._index = index
        self._ads = list(ads)
        self._rebuild_id_map()

    def get_vector(self, ad_id: str) -> np.ndarray:
        if self._index is None:
            raise RuntimeError("vector store is empty; build or load an index first")
        row = self._id_to_row.get(ad_id)
        if row is None:
            raise KeyError(f"ad id not in index: {ad_id}")
        return np.asarray(self._index.reconstruct(row), dtype=np.float32)

    def query(self, vector: np.ndarray, k: int) -> list[ScoredAd]:
        if self._index is None or not self._ads:
            raise RuntimeError("vector store is empty; build or load an index first")
        if k < 1:
            raise ValueError("k must be >= 1")

        query = np.ascontiguousarray(vector, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        faiss.normalize_L2(query)
        top_k = min(k, len(self._ads))
        scores, indices = self._index.search(query, top_k)
        results: list[ScoredAd] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(ScoredAd(ad=self._ads[int(idx)], score=float(score)))
        return results

    def save(self) -> None:
        if self._index is None:
            raise RuntimeError("nothing to save")
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        payload = [ad.to_dict() for ad in self._ads]
        self._sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self) -> None:
        if not self._index_path.exists() or not self._sidecar_path.exists():
            raise FileNotFoundError(
                f"index not found under {self._index_dir} "
                f"(expected {INDEX_FILENAME} and {ADS_SIDECAR_FILENAME})"
            )
        self._index = faiss.read_index(str(self._index_path))
        records = json.loads(self._sidecar_path.read_text(encoding="utf-8"))
        self._ads = [
            Ad(
                id=item["id"],
                domain=item["domain"],
                ad_id=int(item["ad_id"]),
                headline=item["headline"],
                description=item["description"],
                cta=item.get("cta", ""),
                embed_text=item["embed_text"],
            )
            for item in records
        ]
        self._rebuild_id_map()
        if self._index.ntotal != len(self._ads):
            raise ValueError(
                f"index size {self._index.ntotal} does not match sidecar ads {len(self._ads)}"
            )
