"""
Categorize LMArena queries with local Qwen3-32B fp16 via vLLM.

Example:
  .lmarena-env/bin/python lmarena_prep/scripts/generate_categories.py
  .lmarena-env/bin/python lmarena_prep/scripts/generate_categories.py --batch-size 32 --limit 64
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# vLLM/flashinfer JIT looks up the `ninja` binary on PATH, not via pip.
_venv_bin = str(Path(sys.executable).resolve().parent)
os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

import pandas as pd
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data/processed/lmarena/dataset/arena_preference_en_single_turn.parquet"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "data/processed/lmarena/lmarena_query_qwen3_categories.csv"
)

MODEL_NAME = "Qwen/Qwen3-32B"
MAX_NEW_TOKENS = 256
MAX_RETRIES = 3
BATCH_SIZE = 32
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.90
QUERY_LOG_CHARS = 120
QUERY_COLUMN = "query"
CATEGORY_COLUMNS = ["domain", "intent", "commercial_intent"]
CATEGORY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "intent": {"type": "string"},
        "commercial_intent": {"type": "integer", "enum": [0, 1, 2, 3]},
    },
    "required": ["domain", "intent", "commercial_intent"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You classify a single user query for advertising relevance.

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


class QwenClassifier:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_model_len: int = MAX_MODEL_LEN,
        gpu_memory_utilization: float = GPU_MEMORY_UTILIZATION,
    ) -> None:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU is required for Qwen3-32B fp16")
        print(f"Loading {model_name} with vLLM fp16 on {torch.cuda.get_device_name(0)} ...")
        self.llm = LLM(
            model=model_name,
            dtype="float16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
        )
        self.max_model_len = max_model_len
        self.max_prompt_tokens = max_model_len - MAX_NEW_TOKENS
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_NEW_TOKENS,
            structured_outputs=StructuredOutputsParams(
                json=CATEGORY_JSON_SCHEMA,
                disable_additional_properties=True,
            ),
        )
        print(
            "vLLM engine ready. "
            f"max_model_len={max_model_len} max_prompt_tokens={self.max_prompt_tokens}"
        )

    def _apply_chat_template(self, query: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _truncate_query(self, query: str) -> str:
        prompt = self._apply_chat_template(query)
        prompt_ids = self._encode(prompt)
        overflow = len(prompt_ids) - self.max_prompt_tokens
        if overflow <= 0:
            return query

        query_ids = self._encode(query)
        keep = max(0, len(query_ids) - overflow)
        truncated = self.tokenizer.decode(query_ids[:keep], skip_special_tokens=True)

        # Chat-template tokenization can differ slightly from raw query tokens.
        while keep > 0:
            prompt_ids = self._encode(self._apply_chat_template(truncated))
            if len(prompt_ids) <= self.max_prompt_tokens:
                break
            keep = max(0, keep - (len(prompt_ids) - self.max_prompt_tokens))
            truncated = self.tokenizer.decode(query_ids[:keep], skip_special_tokens=True)

        print(
            f"  truncated query from {len(query_ids)} to {keep} tokens "
            f"(prompt {len(self._encode(prompt))} -> "
            f"{len(self._encode(self._apply_chat_template(truncated)))})"
        )
        return truncated

    def _format_prompt(self, query: str) -> str:
        return self._apply_chat_template(self._truncate_query(query))

    def generate_batch(self, queries: list[str]) -> list[str]:
        prompts = [self._format_prompt(q) for q in queries]
        outputs = self.llm.generate(prompts, self.sampling)
        texts = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            texts.append(_strip_thinking(text))
        return texts


def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json_value(text: str):
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_category(payload: dict) -> dict:
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
    return {
        "domain": domain,
        "intent": intent,
        "commercial_intent": commercial_intent,
    }


def _parse_one(raw: str) -> dict:
    payload = _extract_json_value(raw)
    if isinstance(payload, dict) and "results" in payload and payload["results"]:
        payload = payload["results"][0]
    return _normalize_category(payload)


def categorize_queries(queries: list[str], client: QwenClassifier) -> list[dict | None]:
    if not queries:
        return []
    texts = client.generate_batch(queries)
    results: list[dict | None] = [None] * len(queries)
    last_raw = list(texts)
    retry_idx: list[int] = []
    for i, raw in enumerate(texts):
        try:
            results[i] = _parse_one(raw)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            print(f"  parse fail idx={i} ({type(exc).__name__}: {exc}); will retry")
            retry_idx.append(i)

    for attempt in range(1, MAX_RETRIES + 1):
        if not retry_idx:
            break
        print(f"  retrying {len(retry_idx)} queries (attempt {attempt}/{MAX_RETRIES})")
        retry_queries = [queries[i] for i in retry_idx]
        retry_texts = client.generate_batch(retry_queries)
        still_bad: list[int] = []
        for i, raw in zip(retry_idx, retry_texts):
            last_raw[i] = raw
            try:
                results[i] = _parse_one(raw)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                print(f"  parse fail idx={i} again ({type(exc).__name__}: {exc})")
                still_bad.append(i)
        retry_idx = still_bad

    for i in retry_idx:
        print(f"  skipping idx={i} after structured-output parse failure")
        print(f"  raw={last_raw[i][:300]!r}")
        results[i] = None
    return results


def categorize_query(query: str, client: QwenClassifier) -> dict:
    return categorize_queries([query], client)[0]


def _row_already_categorized(row: pd.Series) -> bool:
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
    q_log = " ".join(text.split())
    if len(q_log) > QUERY_LOG_CHARS:
        return q_log[:QUERY_LOG_CHARS] + "..."
    return q_log


def _save_checkpoint(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"  checkpoint saved: {path}")


def categorize_dataframe(
    frame: pd.DataFrame,
    client: QwenClassifier,
    output_path: Path,
    query_column: str = QUERY_COLUMN,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
) -> pd.DataFrame:
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
        results = categorize_queries(pending_queries, client=client)
        for row_idx, query, result in zip(pending_idx, pending_queries, results):
            if result is None:
                print(f"[{labeled + skipped + 1}/{n}] {_truncate(query)} | SKIPPED")
                continue
            out.at[row_idx, "domain"] = result["domain"]
            out.at[row_idx, "intent"] = result["intent"]
            out.at[row_idx, "commercial_intent"] = result["commercial_intent"]
            labeled += 1
            print(
                f"[{labeled + skipped}/{n}] {_truncate(query)} | "
                f"{result['domain']} | {result['intent']} | "
                f"commercial_intent={result['commercial_intent']}"
            )
        pending_idx = []
        pending_queries = []
        _save_checkpoint(out, output_path)

    for i, (idx, row) in enumerate(out.iterrows(), start=1):
        query = "" if pd.isna(row[query_column]) else str(row[query_column])
        if _row_already_categorized(row):
            skipped += 1
            print(f"[{i}/{n}] skip (already categorized) | {_truncate(query)}")
            continue
        pending_idx.append(idx)
        pending_queries.append(query)
        if len(pending_queries) >= batch_size:
            flush()

    flush()
    _save_checkpoint(out, output_path)
    print(f"Saved final CSV: {output_path} (labeled={labeled}, skipped={skipped})")
    return out


def load_input_frame(input_path: Path, query_column: str) -> pd.DataFrame:
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise SystemExit(f"Unsupported input type: {input_path}")
    if query_column not in df.columns:
        raise SystemExit(f"Missing column {query_column!r} in {input_path}")
    df = df.dropna(subset=[query_column]).copy()
    print(f"Loaded {len(df):,} rows from {input_path}")
    return df


def maybe_resume(checkpoint_path: Path, source_df: pd.DataFrame) -> pd.DataFrame:
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

    already = out.apply(_row_already_categorized, axis=1).sum()
    print(f"Resumed {int(already):,} categorized rows; {len(out) - int(already):,} remaining")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query-column", default=QUERY_COLUMN)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    parser.add_argument("--gpu-memory-utilization", type=float, default=GPU_MEMORY_UTILIZATION)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    client = QwenClassifier(
        model_name=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(
        "Qwen/vLLM ready.",
        "model:", args.model,
        "dtype: fp16",
        "batch_size:", args.batch_size,
    )

    source_df = load_input_frame(args.input, args.query_column)
    frame = maybe_resume(args.output, source_df)
    categorized = categorize_dataframe(
        frame,
        client=client,
        output_path=args.output,
        query_column=args.query_column,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    print(categorized[CATEGORY_COLUMNS].head())


if __name__ == "__main__":
    main()
