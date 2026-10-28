"""Characterization tests for CSV row writing and query dataset loading."""
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ad_evaluation.entities import PlacementScore
from src.ad_evaluation.repository import PlacementScoreCsvRepository
from src.ad_generation.old.categories import QueryFrameRepository
from src.ad_integration.repository import QueryDatasetRepository


def _score(i: int) -> PlacementScore:
    return PlacementScore(id=f"q{i}", query="q", ad_id="A::1", position="first", score=3, reason="ok")


class CsvRowWriterTests(unittest.TestCase):
    def test_fresh_write_then_append_without_repeating_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "scores.csv"
            writer = PlacementScoreCsvRepository()
            writer.save([_score(1)], path)
            writer.save([_score(2), _score(3)], path, append=True)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], "id,query,ad_id,position,score,reason")
            self.assertEqual(len(lines), 4)
            self.assertEqual(list(pd.read_csv(path)["id"]), ["q1", "q2", "q3"])

    def test_append_false_overwrites_and_empty_rows_are_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            writer = PlacementScoreCsvRepository()
            writer.save([_score(1), _score(2)], path)
            writer.save([_score(3)], path, append=False)
            writer.save([], path, append=True)
            self.assertEqual(list(pd.read_csv(path)["id"]), ["q3"])

    def test_append_to_missing_file_writes_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scores.csv"
            PlacementScoreCsvRepository().save([_score(1)], path, append=True)
            self.assertEqual(list(pd.read_csv(path)["id"]), ["q1"])


class QueryLoadingTests(unittest.TestCase):
    LOADERS = (QueryDatasetRepository, QueryFrameRepository)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.frame = pd.DataFrame({"id": ["1", "2", "3"], "query": ["a", None, "c"], "extra": [1, 2, 3]})

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_parquet_and_csv_drop_missing_queries(self) -> None:
        parquet = self.root / "q.parquet"
        csv = self.root / "q.csv"
        self.frame.to_parquet(parquet)
        self.frame.to_csv(csv, index=False)
        for loader in self.LOADERS:
            for path in (parquet, csv):
                with self.subTest(loader=loader.__name__, path=path.suffix):
                    loaded = loader().load(path)
                    self.assertEqual(list(loaded["query"]), ["a", "c"])
                    self.assertEqual(list(loaded.columns), ["id", "query", "extra"])

    def test_errors(self) -> None:
        bad_suffix = self.root / "q.txt"
        bad_suffix.write_text("x", encoding="utf-8")
        no_query = self.root / "noq.csv"
        pd.DataFrame({"id": [1]}).to_csv(no_query, index=False)
        for loader in self.LOADERS:
            with self.subTest(loader=loader.__name__):
                with self.assertRaises(FileNotFoundError):
                    loader().load(self.root / "missing.csv")
                with self.assertRaises(ValueError):
                    loader().load(bad_suffix)
                with self.assertRaises(ValueError):
                    loader().load(no_query)


if __name__ == "__main__":
    unittest.main()
