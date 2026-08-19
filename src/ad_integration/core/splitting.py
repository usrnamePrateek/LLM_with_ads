"""Text splitting utilities for segmenting LLM responses into paragraphs or sentences."""
from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ad_integration.config import CHUNK_OVERLAP, CHUNK_SIZE


def normalize_text(text: str) -> str:
    """Normalizes whitespace and newline characters in a string."""
    body = text or ""
    if "\\n" in body:
        body = body.replace("\\n", "\n")
    if "\\t" in body:
        body = body.replace("\\t", "\t")
    return body.replace("\\'", "'").replace('\\"', '"')


_SENTENCE_SPLITTER = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    keep_separator="end",
)


def split_paragraphs(text: str) -> list[str]:
    """Splits a body of text into logical paragraphs using LangChain's RecursiveCharacterTextSplitter."""
    body = normalize_text(text).strip()
    if not body:
        return []
    return [chunk.strip() for chunk in _SENTENCE_SPLITTER.split_text(body) if chunk.strip()]


def join_chunks(chunks: list[str]) -> str:
    return "\n\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip())
