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
)
from src.ad_generation.llm.base import TextGenerator
from src.ad_generation.entities import AdRecord
from src.ad_generation.repository import AdJsonlRepository


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

class GenerateBulkAdsService:
    """Orchestrates the batch generation of synthetic ad creatives for domains in bulk chunks."""
    def __init__(
        self,
        domains: 'src.ad_generation.repository.CsvTaxonomyRepository',
        generator: 'src.ad_generation.llm.ad_generator.VllmBulkAdGenerator',
        writer: AdJsonlRepository,
    ) -> None:
        self._domains = domains
        self._generator = generator
        self._writer = writer

    def run(self) -> list[AdRecord]:
        from src.ad_generation.config import MAX_ADS_PER_PROMPT
        from src.ad_generation.llm.parsing import parse_bulk_ad_creatives
        from collections import defaultdict
        
        domain_counts = self._domains.load_domain_counts()
        existing_counts = self._writer.get_existing_counts()
        
        # Chunk the requests
        chunked_requests = []
        for domain, target_count, topic, subtopic in domain_counts:
            already_generated = existing_counts.get(domain, 0)
            remaining = target_count - already_generated
            if remaining <= 0:
                continue
            
            while remaining > 0:
                chunk = min(remaining, MAX_ADS_PER_PROMPT)
                chunked_requests.append((domain, chunk, topic, subtopic))
                remaining -= chunk
                
        print(f"Total remaining chunks to generate: {len(chunked_requests)}")
        
        ad_id_counter = defaultdict(int, existing_counts)
        all_records = []
        
        batch_size = 50
        for i in range(0, len(chunked_requests), batch_size):
            batch = chunked_requests[i:i + batch_size]
            print(f"Processing chunk batch {i+1} to {min(i+batch_size, len(chunked_requests))}...")
            
            raw_outputs = self._generator.generate_for_domain_counts(
                [(req[0], req[1]) for req in batch]
            )
            batch_records = []
            
            for (domain, _, topic, subtopic), text in zip(batch, raw_outputs):
                try:
                    creatives = parse_bulk_ad_creatives(text)
                except Exception as exc:
                    print(f"Failed for {domain}: {exc}")
                    print(f"Raw output: {text}")
                    continue
                
                for creative in creatives:
                    ad_id_counter[domain] += 1
                    batch_records.append(
                        AdRecord(
                            domain=domain,
                            ad_id=ad_id_counter[domain],
                            headline=creative.headline,
                            description=creative.description,
                            cta=creative.cta,
                            category=topic,
                            subtopic=subtopic,
                        )
                    )
                    
            if batch_records:
                self._writer.append(batch_records)
                all_records.extend(batch_records)
                
        return all_records
