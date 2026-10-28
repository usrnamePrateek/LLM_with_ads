"""Services orchestrating latency and throughput benchmarking for ad retrieval and LLM generation."""
from __future__ import annotations

import gc
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

from src.common.text import strip_thinking
from src.ad_indexing.embeddings.base import EmbeddingEncoder
from src.ad_indexing.embeddings.bge_m3 import BgeM3Encoder
from src.ad_indexing.store.base import VectorStore
from src.ad_indexing.store.faiss_store import FaissVectorStore
from src.common.io import load_queries
from src.common.shared_config import DEFAULT_INDEX_DIR
from src.common.shared_llm import BaseVllmGenerator
from src.speed_comparison.config import (
    CHECKPOINT_EVERY,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_MODEL_LEN,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_QUERIES_PARQUET,
    DEFAULT_RETRIEVAL_DEVICE,
    DEFAULT_SUMMARY_JSON,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    ID_COLUMN,
    QUERY_COLUMN,
)
from src.speed_comparison.entities import (
    QueryLlmTiming,
    QueryRetrievalTiming,
    QuerySpeedComparisonRecord,
    SpeedBenchmarkSummary,
    compute_distribution,
)
from src.speed_comparison.repository import SpeedComparisonRepository


class RetrievalBenchmarkService:
    """Measures precise embedding and vector search latencies for ad retrieval."""

    def __init__(
        self,
        encoder: EmbeddingEncoder,
        store: VectorStore,
        device: str = "cuda",
    ) -> None:
        self._encoder = encoder
        self._store = store
        self._device = device

    def benchmark_query(self, query_id: str, query_text: str, k: int = 1) -> QueryRetrievalTiming:
        """Benchmarks retrieval for a single query with device-appropriate synchronization."""
        clean_text = query_text.strip()
        now_ts = datetime.now(timezone.utc).isoformat()
        if not clean_text:
            return QueryRetrievalTiming(
                id=query_id,
                query=query_text,
                embed_time_ms=0.0,
                search_time_ms=0.0,
                total_retrieval_ms=0.0,
                retrieved_ad_id="",
                retrieved_ad_score=0.0,
                timestamp=now_ts,
            )

        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        vectors = self._encoder.encode_queries([clean_text])
        if self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        embed_time_ms = (t1 - t0) * 1000.0

        t2 = time.perf_counter()
        hits = self._store.query(vectors[0], k)
        t3 = time.perf_counter()
        search_time_ms = (t3 - t2) * 1000.0

        top_id = hits[0].ad.id if hits else ""
        top_score = hits[0].score if hits else 0.0

        return QueryRetrievalTiming(
            id=query_id,
            query=query_text,
            embed_time_ms=embed_time_ms,
            search_time_ms=search_time_ms,
            total_retrieval_ms=embed_time_ms + search_time_ms,
            retrieved_ad_id=top_id,
            retrieved_ad_score=top_score,
            timestamp=now_ts,
        )

    def benchmark_queries(
        self,
        queries: Sequence[tuple[str, str]],
        k: int = 1,
    ) -> list[QueryRetrievalTiming]:
        """Runs retrieval benchmarks sequentially across a list of (id, query) pairs."""
        timings: list[QueryRetrievalTiming] = []
        # Warmup single call
        if queries:
            self._encoder.encode_queries(["warmup query"])

        for q_id, q_text in queries:
            timings.append(self.benchmark_query(q_id, q_text, k=k))
        return timings


class VllmBenchmarkGenerator(BaseVllmGenerator):
    """Subclass of BaseVllmGenerator that records exact prompt and generation latencies."""

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        temperature: float = DEFAULT_TEMPERATURE,
        dtype: str = "auto",
        tensor_parallel_size: int = 1,
    ) -> None:
        super().__init__(
            model_name=model_name,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
        )

    def benchmark_query(self, query_id: str, user_content: str) -> QueryLlmTiming:
        """Generates an LLM response for a single query and measures wall-clock latency."""
        formatted_prompt = self._format_prompt(user_content)
        prompt_token_ids = self._encode(formatted_prompt)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = self._llm.generate([formatted_prompt], self._sampling)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        output_tokens = 0
        raw_text = ""
        if outputs and outputs[0].outputs:
            raw_text = outputs[0].outputs[0].text
            output_tokens = len(outputs[0].outputs[0].token_ids)

        clean_text = strip_thinking(raw_text)
        latency_sec = latency_ms / 1000.0
        tokens_per_sec = (output_tokens / latency_sec) if latency_sec > 0 else 0.0

        return QueryLlmTiming(
            id=query_id,
            query=user_content,
            llm_time_ms=latency_ms,
            prompt_tokens=len(prompt_token_ids),
            output_tokens=output_tokens,
            tokens_per_sec=tokens_per_sec,
            llm_response=clean_text,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def _retrieval_only_summary(
    records: list[QueryRetrievalTiming],
    retrieval_device: str,
    retrieval_wall_clock_sec: float,
) -> SpeedBenchmarkSummary:
    """Builds a summary for a --retrieval-only run; LLM fields are empty or zero."""
    return SpeedBenchmarkSummary(
        sample_size=len(records),
        llm_model="(skipped)",
        embedding_model=EMBEDDING_MODEL_NAME,
        retrieval_device=retrieval_device,
        retrieval_distribution=compute_distribution([r.total_retrieval_ms for r in records]),
        embed_distribution=compute_distribution([r.embed_time_ms for r in records]),
        search_distribution=compute_distribution([r.search_time_ms for r in records]),
        llm_distribution=compute_distribution([]),
        mean_speedup_ratio=0.0,
        median_speedup_ratio=0.0,
        mean_time_delta_ms=0.0,
        pct_retrieval_finished_first=100.0,
        mean_tokens_per_sec=0.0,
        total_retrieval_wall_clock_sec=retrieval_wall_clock_sec,
        total_llm_wall_clock_sec=0.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


class SpeedComparisonOrchestrator:
    """Coordinates two-phase benchmark execution, incremental result writes, and statistical comparison.

    Every run gets a fresh runs/run_<ts>/ folder and is measured in one sitting; there is no resume, so latencies in a
    run are never mixed across sessions. A crashed run keeps its partial CSV (no summary) and is simply rerun.
    """

    def __init__(
        self,
        repository: SpeedComparisonRepository | None = None,
    ) -> None:
        self._repo = repository or SpeedComparisonRepository()

    def run(
        self,
        queries_path: Path = DEFAULT_QUERIES_PARQUET,
        index_dir: Path = DEFAULT_INDEX_DIR,
        output_csv: Path = DEFAULT_OUTPUT_CSV,
        summary_json: Path = DEFAULT_SUMMARY_JSON,
        limit: int | None = 1000,
        llm_model: str = DEFAULT_LLM_MODEL,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        retrieval_device: str = DEFAULT_RETRIEVAL_DEVICE,
        retrieval_only: bool = False,
        checkpoint_every: int = CHECKPOINT_EVERY,
    ) -> SpeedBenchmarkSummary:
        """Executes the full two-phase speed benchmark on queries."""
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = output_csv.parent / f"run_{run_ts}"
        target_dir.mkdir(parents=True, exist_ok=False)

        target_output_csv = target_dir / output_csv.name
        target_summary_json = target_dir / summary_json.name

        print(f"Benchmark run directory: {target_dir}")
        print(f"Output records path: {target_output_csv}")
        print(f"Summary JSON path: {target_summary_json}")

        # 1. Load dataset queries
        frame = load_queries(queries_path, query_column=QUERY_COLUMN)
        if ID_COLUMN not in frame.columns:
            frame[ID_COLUMN] = [str(i) for i in range(len(frame))]
        else:
            frame[ID_COLUMN] = frame[ID_COLUMN].astype(str)

        if limit is not None:
            frame = frame.head(limit).copy()

        total_queries = len(frame)
        print(f"Total queries to benchmark: {total_queries:,}")
        if total_queries == 0:
            raise ValueError(f"no queries to benchmark in {queries_path}")

        query_tuples = [
            (str(row[ID_COLUMN]), str(row[QUERY_COLUMN]))
            for _, row in frame.iterrows()
        ]

        # ── Phase 1: Ad Retrieval Benchmark ────────────────────────────────────
        print(f"\n=== Phase 1: Benchmarking Ad Retrieval (BGE-M3 on {retrieval_device} + FAISS) ===")
        retrieval_timings_dict, t_retrieval_total = self._run_retrieval_phase(
            query_tuples=query_tuples,
            index_dir=index_dir,
            top_k=top_k,
            retrieval_device=retrieval_device,
        )
        print(f"Phase 1 complete in {t_retrieval_total:.2f}s ({total_queries / t_retrieval_total:.1f} queries/s)")

        if retrieval_only:
            print("\n=== --retrieval-only requested: skipping LLM Phase ===")
            retrieval_records = list(retrieval_timings_dict.values())
            summary = _retrieval_only_summary(retrieval_records, retrieval_device, t_retrieval_total)
            retrieval_csv = target_dir / f"retrieval_{output_csv.name}"
            self._repo.save_retrieval_records(retrieval_records, retrieval_csv)
            self._repo.save_summary(summary, target_summary_json)
            self._print_retrieval_report(summary)
            return summary

        # ── Clean GPU memory before loading vLLM ──────────────────────────────
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── Phase 2: LLM Generation Benchmark ─────────────────────────────────
        print(f"\n=== Phase 2: Benchmarking LLM Response Generation ({llm_model}) ===")
        generator = VllmBenchmarkGenerator(
            model_name=llm_model,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature,
        )

        all_records: list[QuerySpeedComparisonRecord] = []
        pending_buffer: list[QuerySpeedComparisonRecord] = []
        wrote_any = False

        t_llm_start = time.perf_counter()
        for idx, (q_id, q_text) in enumerate(query_tuples, start=1):
            llm_timing = generator.benchmark_query(q_id, q_text)
            retrieval_timing = retrieval_timings_dict[q_id]

            retrieval_ms = retrieval_timing.total_retrieval_ms
            llm_ms = llm_timing.llm_time_ms
            speedup = (llm_ms / retrieval_ms) if retrieval_ms > 0 else 0.0

            record = QuerySpeedComparisonRecord(
                id=q_id,
                query=q_text,
                retrieval_time_ms=retrieval_ms,
                embed_time_ms=retrieval_timing.embed_time_ms,
                search_time_ms=retrieval_timing.search_time_ms,
                retrieved_ad_id=retrieval_timing.retrieved_ad_id,
                retrieved_ad_score=retrieval_timing.retrieved_ad_score,
                llm_time_ms=llm_ms,
                prompt_tokens=llm_timing.prompt_tokens,
                output_tokens=llm_timing.output_tokens,
                tokens_per_sec=llm_timing.tokens_per_sec,
                speedup_ratio=speedup,
                time_delta_ms=llm_ms - retrieval_ms,
                retrieval_finished_first=retrieval_ms < llm_ms,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            all_records.append(record)
            pending_buffer.append(record)

            if idx % 10 == 0 or idx == total_queries:
                print(
                    f"  [{idx:,}/{total_queries:,}] "
                    f"Retrieval: {retrieval_ms:.1f}ms | "
                    f"LLM: {llm_ms:.1f}ms ({llm_timing.output_tokens} tok) | "
                    f"Ratio: {speedup:.1f}x"
                )

            # Checkpoint flush
            if len(pending_buffer) >= checkpoint_every or idx == total_queries:
                self._repo.save_records(
                    pending_buffer,
                    target_output_csv,
                    append=wrote_any,
                )
                wrote_any = True
                pending_buffer = []

        t_llm_total = time.perf_counter() - t_llm_start
        print(f"Phase 2 complete in {t_llm_total:.2f}s ({total_queries / t_llm_total:.2f} queries/s)")

        # ── Phase 3: Statistical Aggregation ───────────────────────────────────
        print("\n=== Phase 3: Computing Comparative Statistical Summary ===")
        summary = self._compute_summary(
            records=all_records,
            llm_model=llm_model,
            embedding_model=EMBEDDING_MODEL_NAME,
            retrieval_device=retrieval_device,
            retrieval_wall_clock_sec=t_retrieval_total,
            llm_wall_clock_sec=t_llm_total,
        )
        self._repo.save_summary(summary, target_summary_json)
        self.print_report(summary)
        return summary

    def _run_retrieval_phase(
        self,
        query_tuples: Sequence[tuple[str, str]],
        index_dir: Path,
        top_k: int,
        retrieval_device: str = DEFAULT_RETRIEVAL_DEVICE,
    ) -> tuple[dict[str, QueryRetrievalTiming], float]:
        """Loads index and encoder, executes retrieval benchmark, unloads encoder."""
        store = FaissVectorStore(index_dir)
        store.load()
        encoder = BgeM3Encoder(device=retrieval_device)
        service = RetrievalBenchmarkService(encoder=encoder, store=store, device=retrieval_device)

        if retrieval_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        timings = service.benchmark_queries(query_tuples, k=top_k)
        if retrieval_device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_total = time.perf_counter() - t_start

        # Unload encoder to free VRAM for Phase 2 vLLM
        del encoder
        del service
        del store
        return {item.id: item for item in timings}, t_total

    def _compute_summary(
        self,
        records: list[QuerySpeedComparisonRecord],
        llm_model: str,
        embedding_model: str,
        retrieval_device: str,
        retrieval_wall_clock_sec: float,
        llm_wall_clock_sec: float,
    ) -> SpeedBenchmarkSummary:
        """Aggregates metrics and computes percentiles across all records."""
        retrieval_times = [r.retrieval_time_ms for r in records]
        embed_times = [r.embed_time_ms for r in records]
        search_times = [r.search_time_ms for r in records]
        llm_times = [r.llm_time_ms for r in records]
        speedup_ratios = [r.speedup_ratio for r in records]
        time_deltas = [r.time_delta_ms for r in records]
        tokens_per_sec = [r.tokens_per_sec for r in records]
        retrieval_first_count = sum(1 for r in records if r.retrieval_finished_first)

        return SpeedBenchmarkSummary(
            sample_size=len(records),
            llm_model=llm_model,
            embedding_model=embedding_model,
            retrieval_device=retrieval_device,
            retrieval_distribution=compute_distribution(retrieval_times),
            embed_distribution=compute_distribution(embed_times),
            search_distribution=compute_distribution(search_times),
            llm_distribution=compute_distribution(llm_times),
            mean_speedup_ratio=float(np.mean(speedup_ratios)) if speedup_ratios else 0.0,
            median_speedup_ratio=float(np.median(speedup_ratios)) if speedup_ratios else 0.0,
            mean_time_delta_ms=float(np.mean(time_deltas)) if time_deltas else 0.0,
            pct_retrieval_finished_first=(
                (retrieval_first_count / len(records) * 100.0) if records else 0.0
            ),
            mean_tokens_per_sec=float(np.mean(tokens_per_sec)) if tokens_per_sec else 0.0,
            total_retrieval_wall_clock_sec=retrieval_wall_clock_sec,
            total_llm_wall_clock_sec=llm_wall_clock_sec,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def print_report(summary: SpeedBenchmarkSummary) -> None:
        """Prints a human-readable comparison report table."""
        r_dist = summary.retrieval_distribution
        e_dist = summary.embed_distribution
        s_dist = summary.search_distribution
        l_dist = summary.llm_distribution

        header = f"\n{'=' * 75}\nSPEED COMPARISON BENCHMARK REPORT (N = {summary.sample_size:,})\n{'=' * 75}"
        models_info = (
            f"Embedding Model:   {summary.embedding_model} (Device: {summary.retrieval_device})\n"
            f"LLM Model:         {summary.llm_model}\n"
            f"{'-' * 75}\n"
            f"{'Metric (ms)':<25} {'Mean':>10} {'P50':>10} {'P90':>10} {'P95':>10} {'P99':>10}\n"
            f"{'-' * 75}\n"
            f"{'Ad Retrieval (Total)':<25} {r_dist.mean:>10.2f} {r_dist.p50:>10.2f} {r_dist.p90:>10.2f} {r_dist.p95:>10.2f} {r_dist.p99:>10.2f}\n"
            f"{'  - Query Embedding':<25} {e_dist.mean:>10.2f} {e_dist.p50:>10.2f} {e_dist.p90:>10.2f} {e_dist.p95:>10.2f} {e_dist.p99:>10.2f}\n"
            f"{'  - FAISS Search':<25} {s_dist.mean:>10.2f} {s_dist.p50:>10.2f} {s_dist.p90:>10.2f} {s_dist.p95:>10.2f} {s_dist.p99:>10.2f}\n"
            f"{'LLM Response Gen':<25} {l_dist.mean:>10.2f} {l_dist.p50:>10.2f} {l_dist.p90:>10.2f} {l_dist.p95:>10.2f} {l_dist.p99:>10.2f}\n"
            f"{'-' * 75}\n"
            f"Comparison Findings:\n"
            f"  • Retrieval vs LLM Speedup (Mean):   {summary.mean_speedup_ratio:.1f}x faster\n"
            f"  • Retrieval vs LLM Speedup (Median): {summary.median_speedup_ratio:.1f}x faster\n"
            f"  • Mean Latency Gap:                  {summary.mean_time_delta_ms:.2f} ms\n"
            f"  • Retrieval Finished First:          {summary.pct_retrieval_finished_first:.1f}%\n"
            f"  • Mean LLM Generation Throughput:    {summary.mean_tokens_per_sec:.1f} tokens/s\n"
            f"{'=' * 75}\n"
        )
        print(header)
        print(models_info)

    @staticmethod
    def _print_retrieval_report(summary: SpeedBenchmarkSummary) -> None:
        """Prints a retrieval-only latency report."""
        r_dist = summary.retrieval_distribution
        e_dist = summary.embed_distribution
        s_dist = summary.search_distribution

        header = f"\n{'=' * 75}\nAD RETRIEVAL SPEED BENCHMARK REPORT (N = {summary.sample_size:,})\n{'=' * 75}"
        models_info = (
            f"Embedding Model:   {summary.embedding_model} (Device: {summary.retrieval_device})\n"
            f"Vector Index:      FAISS InnerProduct (CPU)\n"
            f"{'-' * 75}\n"
            f"{'Metric (ms)':<25} {'Mean':>10} {'P50':>10} {'P90':>10} {'P95':>10} {'P99':>10}\n"
            f"{'-' * 75}\n"
            f"{'Ad Retrieval (Total)':<25} {r_dist.mean:>10.2f} {r_dist.p50:>10.2f} {r_dist.p90:>10.2f} {r_dist.p95:>10.2f} {r_dist.p99:>10.2f}\n"
            f"{'  - Query Embedding':<25} {e_dist.mean:>10.2f} {e_dist.p50:>10.2f} {e_dist.p90:>10.2f} {e_dist.p95:>10.2f} {e_dist.p99:>10.2f}\n"
            f"{'  - FAISS Search':<25} {s_dist.mean:>10.2f} {s_dist.p50:>10.2f} {s_dist.p90:>10.2f} {s_dist.p95:>10.2f} {s_dist.p99:>10.2f}\n"
            f"{'-' * 75}\n"
            f"Throughput: {summary.sample_size / summary.total_retrieval_wall_clock_sec:.1f} queries/s\n"
            f"{'=' * 75}\n"
        )
        print(header)
        print(models_info)
