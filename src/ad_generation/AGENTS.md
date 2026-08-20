# `ad_generation` Module

Extracts the Arena dataset, categorizes queries by domain/intent, and generates synthetic ad creatives.

## Resilience Invariant

LLM parsing (`llm/parsing.py`) must handle malformed JSON gracefully. A single row's parse failure must never crash the batch pipeline — log and skip.

## Checkpointing

Category generation processes tens of thousands of rows. The script appends results incrementally and skips already-processed IDs on restart.
