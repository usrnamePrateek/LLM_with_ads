# `ad_evaluation` Module

LLM-as-a-judge evaluation of ad placements. Scores each placement 1–5.

## Masking

`core/masking.py` replaces the inline ad block in the response with `[ad slot]`, then passes the ad text separately in the prompt. The judge sees the ad content but not its rendered position in the response. This forces the judge to evaluate *where* the ad was placed (coherence, disruption) without being influenced by how the ad text visually flows in context.

## Cross-Module Dependencies

Despite being the final pipeline stage, this module imports from:
- `src.ad_integration.core.placement` — `format_ad_block` (used by masking to reconstruct the ad block for replacement)
- `src.ad_generation.llm.parsing` — `extract_json_value` (shared JSON extraction)
- `src.ad_generation.llm.pathutil` — `prepend_venv_bin_to_path`

## Checkpointing

Scoring is slow (batched Llama-70B inference). `ScorePlacementsService` reads existing `placement_scores.csv` on startup and skips already-scored `(id, ad_id, position)` tuples. The `CHECKPOINT_EVERY` config controls flush frequency.
