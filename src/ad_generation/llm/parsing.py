"""Functions for safely parsing structured JSON outputs from LLMs."""
from __future__ import annotations

import json
import re

from src.ad_generation.entities import AdCreative
from src.common.text import strip_thinking


def extract_json_value(raw: str) -> dict | list | str | int | float | bool | None:
    """Extracts and parses a JSON payload from a markdown string (e.g., from ```json blocks)."""
    text = strip_thinking(raw).strip()
    if not text:
        raise json.JSONDecodeError("empty response", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_bulk_ad_creatives(raw: str) -> list[AdCreative]:
    """Validates and parses a raw JSON string into a list of AdCreative objects (unbounded length)."""
    payload = extract_json_value(raw)
    ads = payload.get("ads") if isinstance(payload, dict) else None
    if not isinstance(ads, list) or len(ads) == 0:
        raise ValueError("expected at least 1 ad")
    creatives: list[AdCreative] = []
    for item in ads:
        headline = str(item["headline"]).strip()
        description = str(item["description"]).strip()
        cta = str(item["cta"]).strip()
        if not headline or not description or not cta:
            raise ValueError("ad fields must be non-empty")
        creatives.append(AdCreative(headline=headline, description=description, cta=cta))
    return creatives
