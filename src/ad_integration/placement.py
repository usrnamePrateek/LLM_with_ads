from __future__ import annotations

import re

from src.ad_integration.config import AD_POSITIONS
from src.ad_integration.semantic import insert_ad_after_best_paragraph
from src.ad_integration.splitting import join_chunks, normalize_text, split_paragraphs


def format_ad_block(headline: str, description: str, cta: str) -> str:
    headline_md = normalize_text(headline).strip()
    description_md = normalize_text(description).strip()
    cta_md = normalize_text(cta).strip() or "Learn more"
    cta_md = cta_md.replace("]", "")
    return (
        "---\n\n"
        f"# {headline_md}\n\n"
        f"### {description_md}\n\n"
        f"### [{cta_md}](#)\n\n"
        "---"
    )


def format_llm_response(text: str) -> str:
    return normalize_text(text).strip()


def _split_middle(text: str) -> tuple[str, str]:
    body = normalize_text(text)
    paragraphs = split_paragraphs(body)
    if len(paragraphs) >= 2:
        mid = len(paragraphs) // 2
        return join_chunks(paragraphs[:mid]), join_chunks(paragraphs[mid:])

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


def place_ad(
    llm_response: str,
    ad_block: str,
    position: str,
    chunks: list[str] | None = None,
    best_chunk_index: int | None = None,
) -> str:
    if position not in AD_POSITIONS:
        raise ValueError(f"unknown position {position!r}")
    if position == "first":
        return _wrap([ad_block, format_llm_response(llm_response)])
    if position == "last":
        return _wrap([format_llm_response(llm_response), ad_block])
    if position == "semantic":
        if best_chunk_index is None:
            raise ValueError("semantic placement requires a precomputed best chunk index")
        paragraphs = chunks if chunks is not None else split_paragraphs(llm_response)
        before, after = insert_ad_after_best_paragraph(paragraphs, best_chunk_index)
        return _wrap(
            [
                format_llm_response(before),
                ad_block,
                format_llm_response(after),
            ]
        )
    left, right = _split_middle(llm_response or "")
    return _wrap(
        [
            format_llm_response(left),
            ad_block,
            format_llm_response(right),
        ]
    )
