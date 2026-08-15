from __future__ import annotations

import random


def add_winner_cols(row: dict) -> dict:
    winner = row["winner"]
    if winner == "model_a":
        side = "a"
    elif winner == "model_b":
        side = "b"
    else:
        side = random.Random(row["id"]).choice(["a", "b"])

    if side == "a":
        model = row["model_a"]
        conv = row["conversation_a"]
    else:
        model = row["model_b"]
        conv = row["conversation_b"]

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
