from __future__ import annotations

import re

from generation.config import AD_POSITIONS


def _normalize_text(text: str) -> str:
    body = text or ""
    if "\\n" in body:
        body = body.replace("\\n", "\n")
    if "\\t" in body:
        body = body.replace("\\t", "\t")
    return body.replace("\\'", "'").replace('\\"', '"')


def format_ad_block(headline: str, description: str, cta: str) -> str:
    headline_md = _normalize_text(headline).strip()
    description_md = _normalize_text(description).strip()
    cta_md = _normalize_text(cta).strip() or "Learn more"
    cta_md = cta_md.replace("]", "")
    return (
        "---\n\n"
        f"# {headline_md}\n\n"
        f"### {description_md}\n\n"
        f"### [{cta_md}](#)\n\n"
        "---"
    )


def format_llm_response(text: str) -> str:
    body = _normalize_text(text).strip()
    if not body:
        return ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    if not paragraphs:
        return body
    return "\n\n".join(paragraphs)


def _split_middle(text: str) -> tuple[str, str]:
    body = _normalize_text(text)
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p != ""]
    if len(paragraphs) >= 2:
        mid = len(paragraphs) // 2
        return "\n\n".join(paragraphs[:mid]), "\n\n".join(paragraphs[mid:])

    sentences = re.split(r"(?<=[.!?])\s+", body.strip())
    sentences = [s for s in sentences if s]
    if len(sentences) >= 2:
        mid = len(sentences) // 2
        return " ".join(sentences[:mid]), " ".join(sentences[mid:])

    if not body:
        return "", ""
    mid = max(1, len(body) // 2)
    return body[:mid], body[mid:]


def _wrap(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def place_ad(llm_response: str, ad_block: str, position: str) -> str:
    if position not in AD_POSITIONS:
        raise ValueError(f"unknown position {position!r}")
    if position == "first":
        return _wrap([ad_block, format_llm_response(llm_response)])
    if position == "last":
        return _wrap([format_llm_response(llm_response), ad_block])
    left, right = _split_middle(llm_response or "")
    return _wrap(
        [
            format_llm_response(left),
            ad_block,
            format_llm_response(right),
        ]
    )
