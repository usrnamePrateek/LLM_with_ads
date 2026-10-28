"""Unit tests for the speed_comparison module."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.speed_comparison.entities import (
    LatencyDistribution,
    QueryRetrievalTiming,
    QueryLlmTiming,
    QuerySpeedComparisonRecord,
    SpeedBenchmarkSummary,
    compute_distribution,
)
from src.speed_comparison.repository import SpeedComparisonRepository


class TestSpeedComparisonEntities(unittest.TestCase):
    def test_compute_distribution_empty(self) -> None:
        dist = compute_distribution([])
        self.assertEqual(dist.mean, 0.0)
        self.assertEqual(dist.p50, 0.0)
        self.assertEqual(dist.p99, 0.0)

    def test_compute_distribution_values(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        dist = compute_distribution(values)
        self.assertAlmostEqual(dist.mean, 30.0)
        self.assertAlmostEqual(dist.p50, 30.0)
        self.assertAlmostEqual(dist.min, 10.0)
        self.assertAlmostEqual(dist.max, 50.0)
        self.assertTrue(dist.p90 > dist.p50)
        self.assertTrue(dist.p99 >= dist.p95)

    def test_query_speed_comparison_record(self) -> None:
        record = QuerySpeedComparisonRecord(
            id="test-1",
            query="best shoes",
            retrieval_time_ms=15.5,
            embed_time_ms=14.0,
            search_time_ms=1.5,
            retrieved_ad_id="ad-123",
            retrieved_ad_score=0.88,
            llm_time_ms=310.0,
            prompt_tokens=12,
            output_tokens=65,
            tokens_per_sec=209.67,
            speedup_ratio=20.0,
            time_delta_ms=294.5,
            retrieval_finished_first=True,
        )
        self.assertEqual(record.id, "test-1")
        self.assertTrue(record.retrieval_finished_first)
        self.assertAlmostEqual(record.speedup_ratio, 20.0)


class TestSpeedComparisonRepository(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.TemporaryDirectory()
        self.repo = SpeedComparisonRepository()

    def tearDown(self) -> None:
        self.test_dir.cleanup()

    def test_save_and_load_records(self) -> None:
        output_csv = Path(self.test_dir.name) / "results.csv"
        records = [
            QuerySpeedComparisonRecord(
                id=f"q-{i}",
                query=f"query {i}",
                retrieval_time_ms=10.0 + i,
                embed_time_ms=8.0,
                search_time_ms=2.0 + i,
                retrieved_ad_id=f"ad-{i}",
                retrieved_ad_score=0.9,
                llm_time_ms=200.0 + i * 10,
                prompt_tokens=10,
                output_tokens=50,
                tokens_per_sec=100.0,
                speedup_ratio=20.0,
                time_delta_ms=190.0,
                retrieval_finished_first=True,
            )
            for i in range(3)
        ]
        self.repo.save_records(records, output_csv)
        self.assertTrue(output_csv.exists())

        loaded = pd.read_csv(output_csv)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(list(loaded["id"]), ["q-0", "q-1", "q-2"])

    def test_incremental_writes_append(self) -> None:
        output_csv = Path(self.test_dir.name) / "incremental.csv"
        batch1 = [
            QuerySpeedComparisonRecord(
                id=f"id-{i}",
                query="test",
                retrieval_time_ms=10.0,
                embed_time_ms=8.0,
                search_time_ms=2.0,
                retrieved_ad_id="ad-1",
                retrieved_ad_score=0.9,
                llm_time_ms=200.0,
                prompt_tokens=10,
                output_tokens=50,
                tokens_per_sec=100.0,
                speedup_ratio=20.0,
                time_delta_ms=190.0,
                retrieval_finished_first=True,
            )
            for i in range(2)
        ]
        self.repo.save_records(batch1, output_csv)
        self.assertEqual(list(pd.read_csv(output_csv)["id"]), ["id-0", "id-1"])

        batch2 = [
            QuerySpeedComparisonRecord(
                id="id-2",
                query="test 2",
                retrieval_time_ms=12.0,
                embed_time_ms=9.0,
                search_time_ms=3.0,
                retrieved_ad_id="ad-2",
                retrieved_ad_score=0.85,
                llm_time_ms=220.0,
                prompt_tokens=10,
                output_tokens=55,
                tokens_per_sec=105.0,
                speedup_ratio=18.33,
                time_delta_ms=208.0,
                retrieval_finished_first=True,
            )
        ]
        self.repo.save_records(batch2, output_csv, append=True)
        self.assertEqual(list(pd.read_csv(output_csv)["id"]), ["id-0", "id-1", "id-2"])

    def test_save_summary_json(self) -> None:
        summary_path = Path(self.test_dir.name) / "summary.json"
        dist = LatencyDistribution(10.0, 1.0, 8.0, 10.0, 12.0, 13.0, 14.0, 15.0)
        summary = SpeedBenchmarkSummary(
            sample_size=10,
            llm_model="test-llm",
            embedding_model="test-embed",
            retrieval_device="cuda",
            retrieval_distribution=dist,
            embed_distribution=dist,
            search_distribution=dist,
            llm_distribution=dist,
            mean_speedup_ratio=25.0,
            median_speedup_ratio=24.0,
            mean_time_delta_ms=250.0,
            pct_retrieval_finished_first=100.0,
            mean_tokens_per_sec=120.0,
            total_retrieval_wall_clock_sec=0.5,
            total_llm_wall_clock_sec=2.5,
        )
        self.repo.save_summary(summary, summary_path)
        self.assertTrue(summary_path.exists())

        data = json.loads(summary_path.read_text(encoding="utf-8"))
        self.assertEqual(data["sample_size"], 10)
        self.assertEqual(data["mean_speedup_ratio"], 25.0)
        self.assertEqual(data["retrieval_distribution"]["p50"], 10.0)

    def test_run_folder_structure(self) -> None:
        run_dir = Path(self.test_dir.name) / "run_20261015_000000"
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "speed_comparison_results.csv"
        json_path = run_dir / "speed_comparison_summary.json"

        record = QuerySpeedComparisonRecord(
            id="test-run",
            query="test folder",
            retrieval_time_ms=20.0,
            embed_time_ms=18.0,
            search_time_ms=2.0,
            retrieved_ad_id="ad-1",
            retrieved_ad_score=0.9,
            llm_time_ms=500.0,
            prompt_tokens=10,
            output_tokens=25,
            tokens_per_sec=50.0,
            speedup_ratio=25.0,
            time_delta_ms=480.0,
            retrieval_finished_first=True,
        )
        self.repo.save_records([record], csv_path)
        self.assertTrue(csv_path.exists())
        self.assertEqual(csv_path.parent.name, "run_20261015_000000")


if __name__ == "__main__":
    unittest.main()
