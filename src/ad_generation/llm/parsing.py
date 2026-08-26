"""Functions for safely parsing structured JSON outputs from LLMs."""
from __future__ import annotations

import json
import re

from src.ad_generation.config import CATEGORY_COLUMNS
from src.ad_generation.entities import AdCreative, QueryCategory


def strip_thinking(text: str) -> str:
    """Removes thinking tokens (e.g. <think>...</think>) often produced by reasoning models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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


def parse_ad_creatives(raw: str) -> list[AdCreative]:
    """Validates and parses a raw JSON string into a list of AdCreative objects."""
    payload = extract_json_value(raw)
    ads = payload.get("ads") if isinstance(payload, dict) else None
    if not isinstance(ads, list) or len(ads) != 2:
        raise ValueError("expected exactly 2 ads")
    creatives: list[AdCreative] = []
    for item in ads:
        headline = str(item["headline"]).strip()
        description = str(item["description"]).strip()
        cta = str(item["cta"]).strip()
        if not headline or not description or not cta:
            raise ValueError("ad fields must be non-empty")
        creatives.append(AdCreative(headline=headline, description=description, cta=cta))
    return creatives

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
