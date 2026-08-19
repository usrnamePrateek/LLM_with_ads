"""Service classes that orchestrate data preparation and LLM batch generation workflows."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

from src.ad_generation.arena import add_query_response, add_winner_cols
from src.ad_generation.config import (
    ARENA_DATASET_ID,
    ARENA_KEEP_COLUMNS,
    CATEGORY_BATCH_SIZE,
    CATEGORY_COLUMNS,
    CATEGORY_MAX_RETRIES,
    QUERY_COLUMN,
    QUERY_LOG_CHARS,
)
from src.ad_generation.llm.ad_generator import VllmAdGenerator
from src.ad_generation.llm.base import TextGenerator
from src.ad_generation.entities import AdRecord, QueryCategory
from src.ad_generation.llm.parsing import parse_ad_creatives, parse_category
from src.ad_generation.repository import AdJsonlRepository, QueryFrameRepository, TextDomainRepository


def row_already_categorized(row: pd.Series) -> bool:
    """Checks if a dataframe row already has valid categorizations to skip redundant processing."""
    if any(col not in row.index for col in CATEGORY_COLUMNS):
        return False
    if pd.isna(row["domain"]) or pd.isna(row["intent"]) or pd.isna(row["commercial_intent"]):
        return False
    domain = str(row["domain"]).strip()
    intent = str(row["intent"]).strip()
    if not domain or not intent or domain.lower() == "nan" or intent.lower() == "nan":
        return False
    try:
        ci = int(row["commercial_intent"])
    except (TypeError, ValueError):
        return False
    return ci in (0, 1, 2, 3)


def _truncate(text: str) -> str:
    """Truncates strings for cleaner console logging."""
    q_log = " ".join(text.split())
    if len(q_log) > QUERY_LOG_CHARS:
        return q_log[:QUERY_LOG_CHARS] + "..."
    return q_log


class CategorizeQueriesService:
    """Orchestrates the batch processing of user queries to determine their domain and commercial intent via an LLM."""
    def __init__(
        self,
        generator: TextGenerator,
        repository: QueryFrameRepository,
        max_retries: int = CATEGORY_MAX_RETRIES,
    ) -> None:
        self._generator = generator
        self._repository = repository
        self._max_retries = max_retries

    def categorize_queries(self, queries: list[str]) -> list[QueryCategory | None]:
        """Sends a batch of queries to the LLM and parses the structured responses, handling retries on parse failures."""
        if not queries:
            return []
        texts = self._generator.generate(queries)
        results: list[QueryCategory | None] = [None] * len(queries)
        last_raw = list(texts)
        retry_idx: list[int] = []
        for i, raw in enumerate(texts):
            try:
                results[i] = parse_category(raw)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                print(f"  parse fail idx={i} ({type(exc).__name__}: {exc}); will retry")
                retry_idx.append(i)

        for attempt in range(1, self._max_retries + 1):
            if not retry_idx:
                break
            print(f"  retrying {len(retry_idx)} queries (attempt {attempt}/{self._max_retries})")
            retry_queries = [queries[i] for i in retry_idx]
            retry_texts = self._generator.generate(retry_queries)
            still_bad: list[int] = []
            for i, raw in zip(retry_idx, retry_texts):
                last_raw[i] = raw
                try:
                    results[i] = parse_category(raw)
                except (json.JSONDecodeError, ValueError, TypeError) as exc:
                    print(f"  parse fail idx={i} again ({type(exc).__name__}: {exc})")
                    still_bad.append(i)
            retry_idx = still_bad

        for i in retry_idx:
            print(f"  skipping idx={i} after structured-output parse failure")
            print(f"  raw={last_raw[i][:300]!r}")
            results[i] = None
        return results

    def run(
        self,
        frame: pd.DataFrame,
        output_path: Path,
        query_column: str = QUERY_COLUMN,
        batch_size: int = CATEGORY_BATCH_SIZE,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Main execution loop that chunks queries into batches, dispatches to the LLM, and checkpoints results to disk."""
        out = frame.copy()
        if limit is not None:
            out = out.head(limit).copy()
        for col in CATEGORY_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA

        n = len(out)
        pending_idx: list = []
        pending_queries: list[str] = []
        labeled = 0
        skipped = 0

        def flush() -> None:
            nonlocal pending_idx, pending_queries, labeled
            if not pending_idx:
                return
            print(
                f"Running vLLM batch of {len(pending_queries)} "
                f"({labeled + skipped + 1}-{labeled + skipped + len(pending_queries)}/{n})"
            )
            results = self.categorize_queries(pending_queries)
            for row_idx, query, result in zip(pending_idx, pending_queries, results):
                if result is None:
                    print(f"[{labeled + skipped + 1}/{n}] {_truncate(query)} | SKIPPED")
                    continue
                out.at[row_idx, "domain"] = result.domain
                out.at[row_idx, "intent"] = result.intent
                out.at[row_idx, "commercial_intent"] = result.commercial_intent
                labeled += 1
                print(
                    f"[{labeled + skipped}/{n}] {_truncate(query)} | "
                    f"{result.domain} | {result.intent} | "
                    f"commercial_intent={result.commercial_intent}"
                )
            pending_idx = []
            pending_queries = []
            self._repository.save_csv(out, output_path)

        for i, (idx, row) in enumerate(out.iterrows(), start=1):
            query = "" if pd.isna(row[query_column]) else str(row[query_column])
            if row_already_categorized(row):
                skipped += 1
                print(f"[{i}/{n}] skip (already categorized) | {_truncate(query)}")
                continue
            pending_idx.append(idx)
            pending_queries.append(query)
            if len(pending_queries) >= batch_size:
                flush()

        flush()
        self._repository.save_csv(out, output_path)
        print(f"Saved final CSV: {output_path} (labeled={labeled}, skipped={skipped})")
        already = out.apply(row_already_categorized, axis=1).sum()
        print(f"Categorized rows: {int(already):,}")
        return out


class GenerateAdsService:
    """Orchestrates the batch generation of synthetic ad creatives for a given list of domains."""
    def __init__(
        self,
        domains: TextDomainRepository,
        generator: VllmAdGenerator,
        writer: AdJsonlRepository,
    ) -> None:
        self._domains = domains
        self._generator = generator
        self._writer = writer

    def run(self) -> list[AdRecord]:
        """Loads domains, generates ad copy via an LLM, parses the creatives, and persists them."""
        domains = self._domains.load()
        raw_outputs = self._generator.generate_for_domains(domains)
        records: list[AdRecord] = []
        for domain, text in zip(domains, raw_outputs):
            try:
                creatives = parse_ad_creatives(text)
            except Exception as exc:
                print(f"Failed for {domain}: {exc}")
                print(f"Raw output: {text}")
                continue
            for ad_id, creative in enumerate(creatives, start=1):
                records.append(
                    AdRecord(
                        domain=domain,
                        ad_id=ad_id,
                        headline=creative.headline,
                        description=creative.description,
                        cta=creative.cta,
                    )
                )
        self._writer.save(records)
        return records


class PrepareArenaDatasetService:
    """Orchestrates the download, filtering, and normalization of the raw huggingface LMArena dataset."""
    def run(self, output_dir: Path) -> None:
        """Downloads the dataset, applies structural transformations to flatten conversations, and saves it to Parquet."""
        load_dotenv()
        token = os.getenv("HF_TOKEN")
        if not token:
            raise SystemExit("Set HF_TOKEN in .env (https://huggingface.co/settings/tokens)")

        print(f"Loading {ARENA_DATASET_ID} ...")
        ds = load_dataset(ARENA_DATASET_ID, token=token)
        train = ds["train"]
        print(f"  start: {len(train):,}")

        train = train.filter(lambda x: not x["is_code"])
        print(f"  after drop is_code: {len(train):,}")

        train = train.filter(lambda x: x["language"] == "en")
        print(f"  after English only: {len(train):,}")

        train = train.filter(lambda x: x["winner"] != "both_bad")
        print(f"  after drop both_bad: {len(train):,}")

        train = train.map(add_winner_cols)
        print("  added model + conversation")

        train = train.filter(lambda x: len(x["conversation"]) == 2)
        print(f"  after conversation length == 2: {len(train):,}")

        train = train.map(add_query_response)
        print("  added query + llm_response")

        cols = [c for c in ARENA_KEEP_COLUMNS if c in train.column_names]
        train = train.select_columns(cols)

        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "arena_preference_en_single_turn.parquet"
        hf_dir = output_dir / "hf"

        train.to_parquet(str(parquet_path))
        train.save_to_disk(str(hf_dir))

        print(f"Saved {len(train):,} rows")
        print(f"  parquet: {parquet_path}")
        print(f"  hf disk: {hf_dir}")
