"""Legacy query categorization (Qwen3-32B): config, entity, prompt, parser, checkpoint repository and service.

Not part of the current pipeline. Bulk ad generation uses taxonomy sub-topics instead of query categories.
Kept so the original categories CSV can be reproduced; see ../README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ad_generation.llm.base import TextGenerator
from src.ad_generation.llm.parsing import extract_json_value
from src.common.io import load_queries
from src.common.shared_config import REPO_ROOT

DEFAULT_CATEGORIES_OUTPUT = REPO_ROOT / "data/processed/lmarena/old/lmarena_query_qwen3_categories.csv"

QUERY_COLUMN = "query"
CATEGORY_COLUMNS = ["domain", "intent", "commercial_intent"]
QUERY_LOG_CHARS = 120

CATEGORY_MODEL_NAME = "Qwen/Qwen3-32B"
CATEGORY_MAX_NEW_TOKENS = 256
CATEGORY_MAX_RETRIES = 3
CATEGORY_BATCH_SIZE = 32
CATEGORY_MAX_MODEL_LEN = 32768
CATEGORY_GPU_MEMORY_UTILIZATION = 0.90


@dataclass(frozen=True)
class QueryCategory:
    """Represents the parsed categorization of a user query."""
    domain: str
    intent: str
    commercial_intent: int


CATEGORY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "intent": {"type": "string"},
        "commercial_intent": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": list(CATEGORY_COLUMNS),
    "additionalProperties": False,
}

CATEGORY_SYSTEM_PROMPT = """You classify a single user query for advertising relevance.

Return ONLY a JSON object with exactly these fields:
{"domain": "...", "intent": "...", "commercial_intent": 0}

Do not return any other fields. Do not return explanations.

Fields:
- domain: concise reusable topic label (e.g. "Sports & Fitness", "Technology", "Travel")
- intent: concise reusable user-intent label (e.g. "Information Seeking", "Product Research", "Purchase")
- commercial_intent: integer 0, 1, 2, or 3

Primary goal: USER INTENT and COMMERCIAL INTENT, not merely the topic.
Do not infer commercial intent just because the topic has products associated with it.
Informational questions about commercial topics still get commercial_intent = 0.

commercial_intent scale:
0 = No commercial intent. Pure information, explanation, creative content, general knowledge, personal advice. No obvious product/service/business opportunity.
1 = Weak commercial intent. Possible connection to a product/service, but the user is not clearly researching or considering a purchase.
2 = Moderate commercial intent. Researching, comparing, evaluating, or seeking recommendations that could reasonably lead to a purchase or paid product/service.
3 = Strong commercial intent. Explicitly wants to buy, book, order, hire, subscribe, find a provider, or take another clear commercial action.

Label style:
- Keep domain and intent concise, consistent, and reusable across a dataset.
- Prefer "Sports & Fitness" over "Sports, Exercise, Physical Activity, and Fitness".
- Prefer "Product Research" over "Researching potential products to purchase".

Examples:
Query: "What is quantum entanglement?"
{"domain": "Science", "intent": "Information Seeking", "commercial_intent": 0}

Query: "What is an iPhone?"
{"domain": "Technology", "intent": "Information Seeking", "commercial_intent": 0}

Query: "Which iPhone should I buy?"
{"domain": "Technology", "intent": "Product Research", "commercial_intent": 2}

Query: "Where can I buy an iPhone?"
{"domain": "Technology", "intent": "Purchase", "commercial_intent": 3}

Query: "What are the best running shoes for beginners?"
{"domain": "Sports & Fitness", "intent": "Product Research", "commercial_intent": 2}

Query: "Where can I buy running shoes?"
{"domain": "Sports & Fitness", "intent": "Purchase", "commercial_intent": 3}

Query: "Compare Nike and Adidas running shoes"
{"domain": "Sports & Fitness", "intent": "Product Comparison", "commercial_intent": 2}

Query: "Find a good hotel in Paris"
{"domain": "Travel", "intent": "Travel Planning", "commercial_intent": 3}

Query: "Explain how airplanes fly"
{"domain": "Science", "intent": "Information Seeking", "commercial_intent": 0}

Query: "What are some good movies to watch?"
{"domain": "Entertainment", "intent": "Recommendation", "commercial_intent": 1}

Query: "Write me a poem about the ocean"
{"domain": "Creative Writing", "intent": "Creative", "commercial_intent": 0}

Recommendations/comparisons about products/services generally get 2.
Explicit purchasing/booking/hiring/finding-provider requests generally get 3.
Creative writing and general knowledge generally get 0.
"""


def parse_category(raw: str) -> QueryCategory:
    """Validates and parses a raw JSON string into a QueryCategory domain object."""
    payload = extract_json_value(raw)
    if isinstance(payload, dict) and "results" in payload and payload["results"]:
        payload = payload["results"][0]
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    extra = set(payload) - set(CATEGORY_COLUMNS)
    missing = [k for k in CATEGORY_COLUMNS if k not in payload]
    if extra or missing:
        raise ValueError(f"unexpected fields extra={sorted(extra)} missing={missing}")

    domain = str(payload["domain"]).strip()
    intent = str(payload["intent"]).strip()
    commercial_intent = payload["commercial_intent"]
    if isinstance(commercial_intent, str) and commercial_intent.strip().isdigit():
        commercial_intent = int(commercial_intent.strip())
    if not isinstance(commercial_intent, int) or isinstance(commercial_intent, bool):
        raise ValueError("commercial_intent must be an integer")
    if commercial_intent not in (0, 1, 2, 3):
        raise ValueError("commercial_intent must be 0-3")
    if not domain or not intent:
        raise ValueError("domain and intent must be non-empty")
    return QueryCategory(domain=domain, intent=intent, commercial_intent=commercial_intent)


class QueryFrameRepository:
    """Repository for reading and checkpointing pandas DataFrames containing user queries."""

    def load(self, path: Path, query_column: str = QUERY_COLUMN) -> pd.DataFrame:
        """Loads a Parquet or CSV file containing queries, filtering out rows with missing query data."""
        return load_queries(path, query_column)

    def merge_checkpoint(self, checkpoint_path: Path, source_df: pd.DataFrame) -> pd.DataFrame:
        """Merges previously categorized query rows from a CSV checkpoint to resume interrupted batch generation jobs."""
        if not checkpoint_path.exists():
            return source_df

        resumed = pd.read_csv(checkpoint_path)
        print(f"Loaded checkpoint ({len(resumed):,} rows): {checkpoint_path}")
        out = source_df.copy()
        for col in CATEGORY_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA

        label_cols = [c for c in CATEGORY_COLUMNS if c in resumed.columns]
        if not label_cols:
            return out

        if "id" in out.columns and "id" in resumed.columns:
            ckpt = resumed[["id"] + label_cols].drop_duplicates(subset=["id"], keep="last")
            out = out.drop(columns=label_cols).merge(ckpt, on="id", how="left")
        else:
            n = min(len(out), len(resumed))
            for col in label_cols:
                out.iloc[:n, out.columns.get_loc(col)] = resumed[col].iloc[:n].to_numpy()
        return out

    def save_csv(self, frame: pd.DataFrame, path: Path) -> None:
        """Saves a pandas DataFrame to CSV, creating parent directories if needed."""
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"  checkpoint saved: {path}")


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
