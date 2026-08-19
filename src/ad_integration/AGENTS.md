# Agent Instructions: `ad_integration` Module

## Purpose
This module handles the core "ad placement" logic. It matches a user query to the most relevant synthetic advertisement and injects that ad into an LLM response.

## Responsibilities
- Query-to-Ad matching (assigning top ads based on vector similarity).
- Structurally modifying LLM response strings to inject markdown ad blocks.
- Semantic parsing (splitting text into paragraphs) for intelligent placement.

## Dependencies
- **Allowed**: `src.common`, `src.ad_indexing` (to query the FAISS index).
- **Forbidden**: Do not import from `src.ad_evaluation` to avoid circular dependencies.

## Important Interfaces
- `assign_top_ads.py`: Entrypoint for matching queries to ads.
- `place_ads.py`: Entrypoint for text injection.
- `placement.py` (`place_ad`): Core pure function that accepts an LLM response, an ad block, and a target position (`first`, `middle`, `last`, `semantic`) and returns the combined markdown string.
- `semantic.py`: Analyzes paragraphs and determines the best injection boundary using cosine similarity against the query.

## Design Rules
- **Pure Functions**: Keep string manipulation logic (`placement.py`, `splitting.py`) as pure, deterministic functions. They should not rely on external state, networking, or the file system.
- **Fail Gracefully**: LLM outputs can be malformed, empty, or lack punctuation. Paragraph splitting and injection logic must handle edge cases gracefully without throwing exceptions.

## Data Flow
1. Reads `arena_preference_en_single_turn.parquet` (queries/responses) and FAISS index.
2. `assign_top_ads.py` outputs a mapping of queries to ads (`query_top1_ads.csv`).
3. `place_ads.py` reads the mapping and writes the injected texts to `query_ad_positions.csv`.

## Testing
- Ensure high test coverage for `placement.py` and `splitting.py`. Pass empty strings, extremely long single paragraphs, and strings with weird whitespace to ensure robustness.
