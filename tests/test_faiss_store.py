"""FaissVectorStore save/load must round-trip every Ad field."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.ad_indexing.entities import Ad
from src.ad_indexing.store.faiss_store import FaissVectorStore


def _ad(ad_id: int, category: str | None) -> Ad:
    return Ad.from_record(
        {"domain": "A", "ad_id": ad_id, "headline": f"h{ad_id}", "description": "d", "cta": "c", "category": category}
    )


class FaissStoreRoundTripTests(unittest.TestCase):
    def test_save_then_load_keeps_all_fields_including_category(self) -> None:
        ads = [_ad(1, "Travel"), _ad(2, None)]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            store = FaissVectorStore(Path(tmp))
            store.upsert([ad.id for ad in ads], vectors, ads)
            store.save()
            loaded = FaissVectorStore(Path(tmp))
            loaded.load()
            hits = loaded.query(np.array([1.0, 0.0], dtype=np.float32), 2)
        self.assertEqual({hit.ad for hit in hits}, set(ads))
        self.assertEqual(hits[0].ad.category, "Travel")

    def test_old_sidecar_without_category_still_loads(self) -> None:
        ad = _ad(1, "Travel")
        with tempfile.TemporaryDirectory() as tmp:
            store = FaissVectorStore(Path(tmp))
            store.upsert([ad.id], np.array([[1.0, 0.0]], dtype=np.float32), [ad])
            store.save()
            sidecar = Path(tmp) / "ads.json"
            records = json.loads(sidecar.read_text(encoding="utf-8"))
            for record in records:
                record.pop("category")
            sidecar.write_text(json.dumps(records), encoding="utf-8")
            loaded = FaissVectorStore(Path(tmp))
            loaded.load()
            hit = loaded.query(np.array([1.0, 0.0], dtype=np.float32), 1)[0]
        self.assertIsNone(hit.ad.category)
        self.assertEqual(hit.ad.headline, "h1")


if __name__ == "__main__":
    unittest.main()
