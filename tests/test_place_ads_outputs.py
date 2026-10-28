"""place_ads must never destroy its previous outputs unless a run completes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.ad_integration.repository import PlacedAdCsvRepository, QueryDatasetRepository, SemanticParagraphCsvRepository
from src.ad_integration.services import PlaceAdsInResponsesService


class StubEncoder:
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class StubStore:
    def __init__(self, fail_on: str | None = None) -> None:
        self._fail_on = fail_on

    def get_vector(self, ad_id: str) -> np.ndarray:
        if ad_id == self._fail_on:
            raise RuntimeError("simulated crash")
        return np.ones(2, dtype=np.float32)


class PlaceAdsOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.dataset = root / "dataset.csv"
        pd.DataFrame(
            {"id": ["q1", "q2"], "query": ["a?", "b?"], "llm_response": ["Answer one.", "Answer two."]}
        ).to_csv(self.dataset, index=False)
        self.ads = root / "top_ads.csv"
        pd.DataFrame(
            {"id": ["q1", "q2"], "ad_id": ["A::1", "A::2"], "ad_domain": ["A", "A"],
             "headline": ["h1", "h2"], "description": ["d1", "d2"], "cta": ["c", "c"]}
        ).to_csv(self.ads, index=False)
        self.output = root / "positions.csv"
        self.semantic = root / "semantic.csv"
        self.output.write_text("previous run\n", encoding="utf-8")
        self.semantic.write_text("previous run\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, store: StubStore):
        writer = PlacedAdCsvRepository()
        return PlaceAdsInResponsesService(
            queries=QueryDatasetRepository(), assignments=writer, writer=writer,
            semantic_writer=SemanticParagraphCsvRepository(), encoder=StubEncoder(), store=store,
        ).run(self.dataset, self.ads, self.output, semantic_output_path=self.semantic)

    def test_crash_keeps_previous_outputs(self) -> None:
        with self.assertRaises(RuntimeError):
            self._run(StubStore(fail_on="A::2"))
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous run\n")
        self.assertEqual(self.semantic.read_text(encoding="utf-8"), "previous run\n")

    def test_successful_run_replaces_outputs_and_leaves_no_partials(self) -> None:
        rows = self._run(StubStore())
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(pd.read_csv(self.output)), 8)
        self.assertEqual(len(pd.read_csv(self.semantic)), 2)
        self.assertEqual(sorted(p.name for p in self.output.parent.glob("*.partial.csv")), [])


if __name__ == "__main__":
    unittest.main()
