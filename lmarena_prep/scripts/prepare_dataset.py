"""
Prepare Arena Human Preference data for downstream use.

Pipeline:
  1. Drop is_code rows
  2. Keep English only
  3. Choose winning model (drop both_bad; random side on tie)
  4. Keep conversations with length == 2 (single turn)
  5. Extract query / llm_response
  6. Save the final dataset
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
import os


DATASET_ID = "lmarena-ai/arena-human-preference-140k"
KEEP_COLUMNS = ["id", "model", "query", "llm_response", "timestamp"]


def add_winner_cols(row: dict) -> dict:
    winner = row["winner"]
    if winner == "model_a":
        side = "a"
    elif winner == "model_b":
        side = "b"
    else:  # tie — deterministic random choice from id
        side = random.Random(row["id"]).choice(["a", "b"])

    if side == "a":
        model = row["model_a"]
        conv = row["conversation_a"]
    else:
        model = row["model_b"]
        conv = row["conversation_b"]

    # Normalize schema (conversation_b has optional num_tokens)
    conversation = [{"role": m["role"], "content": m["content"]} for m in conv]
    return {"model": model, "conversation": conversation}


def _text_from_msg(msg: dict) -> str:
    return "".join(
        part["text"] or "" for part in msg["content"] if part["type"] == "text"
    )


def add_query_response(row: dict) -> dict:
    query = ""
    llm_response = ""
    for msg in row["conversation"]:
        if msg["role"] == "user":
            query = _text_from_msg(msg)
        elif msg["role"] == "assistant":
            llm_response = _text_from_msg(msg)
    return {"query": query, "llm_response": llm_response}


def prepare(output_dir: Path) -> None:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN in .env (https://huggingface.co/settings/tokens)")

    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, token=token)
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

    cols = [c for c in KEEP_COLUMNS if c in train.column_names]
    train = train.select_columns(cols)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "arena_preference_en_single_turn.parquet"
    hf_dir = output_dir / "hf"

    train.to_parquet(str(parquet_path))
    train.save_to_disk(str(hf_dir))

    print(f"Saved {len(train):,} rows")
    print(f"  parquet: {parquet_path}")
    print(f"  hf disk: {hf_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for parquet + HF disk save (default: data/processed)",
    )
    args = parser.parse_args()
    prepare(args.output_dir)


if __name__ == "__main__":
    main()
