# `ad_evaluation` Module

LLM-as-a-judge evaluation of ad placements and ad preference. Scores each placement 1–5, and determines pairwise ad preference with confidence levels (low, medium, high).

## Masking

`core/masking.py` replaces the inline ad block in the response with `[ad slot]`, then passes the ad text separately in the prompt. The judge sees the ad content but not its rendered position in the response. This forces the judge to evaluate *where* the ad was placed (coherence, disruption) without being influenced by how the ad text visually flows in context.

## Pairwise Preference Evaluation

The `EvaluateAdPreferenceService` evaluates an assigned ad against all other ads in the same category. It tests each pair bidirectionally (both A vs B and B vs A) to eliminate positional bias, using a dedicated `VllmPreferenceJudge` to output a winner and a confidence level (low/medium/high).

## Cross-Module Dependencies

Despite being the final pipeline stage, this module imports from:
- `src.ad_integration.core.placement` — `format_ad_block` (used by masking to reconstruct the ad block for replacement)
- `src.ad_generation.llm.parsing` — `extract_json_value` (shared JSON extraction)
- `src.ad_generation.llm.pathutil` — `prepend_venv_bin_to_path`
- `src.ad_indexing.repository` — `JsonlAdRepository` (for loading the ad catalog to group by category)

## Checkpointing

Scoring is slow (batched Llama-70B inference). `ScorePlacementsService` reads existing `placement_scores.csv` on startup and skips already-scored `(id, ad_id, position)` tuples. The `CHECKPOINT_EVERY` config controls flush frequency.
