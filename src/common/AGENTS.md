# `common` Module

Shared infrastructure layer. Must not contain business logic.

## Dependency Warning

`shared_llm.py` imports `strip_thinking` from `src.ad_generation.llm.parsing`. This is the only cross-module import in `common`. Do not add more — it should ideally be refactored into `common` itself.

## `BaseVllmGenerator`

All LLM generators inherit from this class. It handles:
- vLLM instantiation and GPU memory config
- Chat template application with `enable_thinking=False` fallback
- Token truncation: trims user content to fit `max_model_len - max_new_tokens`

Changes to truncation logic affect every generator in the project (`VllmCategoryGenerator`, `VllmAdGenerator`, `VllmPlacementJudge`).
