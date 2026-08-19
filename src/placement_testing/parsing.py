from __future__ import annotations

from src.lmarena_prep.parsing import extract_json_value


def parse_placement_score(raw: str) -> tuple[int, str]:
    payload = extract_json_value(raw)
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    score = payload.get("score")
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    if not isinstance(score, int) or isinstance(score, bool) or score not in (1, 2, 3, 4, 5):
        raise ValueError(f"score must be an integer 1-5, got {score!r}")
    reason = str(payload.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason must be non-empty")
    return score, reason
