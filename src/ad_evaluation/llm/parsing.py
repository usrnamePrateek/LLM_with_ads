"""Logic for extracting and validating LLM judge outputs."""
from __future__ import annotations

from src.ad_generation.llm.parsing import extract_json_value


def parse_placement_score(raw: str) -> tuple[int, str]:
    """Parses a raw LLM response into an integer score and string rationale."""
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


def parse_pairwise_preference(raw: str) -> tuple[str, str]:
    """Parses a raw LLM response into a winning ad ID ('ad_1' or 'ad_2') and confidence level."""
    payload = extract_json_value(raw)
    if not isinstance(payload, dict):
        raise ValueError("response is not a JSON object")
    winner = payload.get("winner_ad")
    if winner not in ("ad_1", "ad_2"):
        raise ValueError(f"winner_ad must be 'ad_1' or 'ad_2', got {winner!r}")
    confidence = str(payload.get("confidence", "")).strip().lower()
    if confidence not in ("low", "medium", "high"):
        raise ValueError(f"confidence must be 'low', 'medium', or 'high', got {confidence!r}")
    return winner, confidence
