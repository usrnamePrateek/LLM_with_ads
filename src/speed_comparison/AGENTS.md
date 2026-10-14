# `speed_comparison` Module

Benchmarks and compares query-time ad retrieval latency against LLM response generation latency.

## Responsibilities

1. **Retrieval Latency Measurement**: Coordinates `BgeM3Encoder` (dense query embedding) and `FaissVectorStore` (FAISS inner product search) to measure embedding time, vector search time, and total retrieval latency per query.
2. **LLM Generation Latency Measurement**: Inherits `BaseVllmGenerator` from `src/common/shared_llm.py` via `VllmBenchmarkGenerator` to measure exact wall-clock response generation latency, token counts, and token generation throughput (tokens/sec).
3. **Statistical Aggregation**: Computes comparative metrics including speedup ratio (`llm_time / retrieval_time`), time delta (`llm_time - retrieval_time`), and full latency distributions (mean, std, min, p50, p90, p95, p99, max).

## Two-Phase Execution Architecture

Because both `vLLM` and `BgeM3Encoder` require GPU resources when running on CUDA, running them concurrently on a single GPU can cause CUDA out-of-memory errors:
- **Phase 1 (Retrieval)**: Loads `FaissVectorStore` and `BgeM3Encoder` on `--retrieval-device` (`cuda` by default, or `cpu`), runs retrieval on all queries, logs timings, then unloads the encoder and invokes `torch.cuda.empty_cache()` and `gc.collect()`.
- **Phase 2 (LLM Generation)**: Instantiates `VllmBenchmarkGenerator` on GPU, generates responses, measures generation times, and periodically checkpoints results.
- **Phase 3 (Analysis & Report)**: Merges records, computes distributions, exports CSV + JSON, and prints a formatted terminal report.

Note: If `--retrieval-device cpu` is specified, BGE-M3 encodes on CPU (zero GPU VRAM usage), while FAISS always executes on CPU via `faiss-cpu`.

## Conventions

- **Checkpointing**: The orchestrator checks for existing query IDs in the output CSV and appends incrementally every `checkpoint_every` queries to prevent data loss.
- **Serialization**: All entities are serialized using `dataclasses.asdict()`. No custom `to_dict()` methods.
- **Config Independence**: Module-specific constants in `config.py` are declared directly rather than derived from other packages.
