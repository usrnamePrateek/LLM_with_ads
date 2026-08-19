# Agent Instructions: `ad_generation` Module

## Purpose
This module handles the foundational data preparation for the project. It extracts the raw human-preference dataset, uses LLMs to categorize the queries (domain/intent), and uses LLMs to generate synthetic advertisements based on those domains.

## Responsibilities
- Parsing and filtering the LMSYS Chatbot Arena Parquet dataset.
- Extracting clean queries and responses.
- Running LLM batch generation to categorize queries.
- Running LLM batch generation to create synthetic ad copy (headlines, descriptions, CTAs).

## Dependencies
- **Allowed**: `src.common`
- **Forbidden**: Do not depend on `ad_integration`, `ad_indexing`, or `ad_evaluation`. This is the first module in the pipeline and must remain independent.

## Important Interfaces
- `prepare_dataset.py`: Cleans and filters the huggingface dataset.
- `generate_categories.py`: Classifies queries for domain and intent.
- `generate_ads.py`: Generates the actual ad creatives.
- `VllmCategoryGenerator` & `VllmAdGenerator`: Wrappers around `src.common.shared_llm.BaseVllmGenerator`.

## Design Rules
- **Batch Processing**: Dataset manipulation should be heavily vectorized using `pandas`. Avoid looping over dataframe rows natively in python.
- **Resilience**: The LLM parsing logic must handle edge cases where the LLM fails to return valid JSON. Do not crash the entire batch pipeline for a single row's parsing error; skip or log the error and continue.

## Data Flow
1. Input: Remote Parquet dataset (`lmarena-ai/arena-human-preference-140k`).
2. Data processing output: Filtered local Parquet.
3. Category generator output: `domains_sorted.csv`.
4. Ad generator output: `ads.jsonl`.

## Performance Considerations
- **Checkpointing**: The category generation script processes tens of thousands of rows. It must maintain a checkpointing/append-mode architecture. If the process is interrupted, it must be able to resume without starting from zero.
- **GPU Memory**: vLLM holds a significant amount of GPU memory. Ensure the `gpu_memory_utilization` is tuned correctly in `src.common.shared_config` to prevent OOM errors when running large batches.
