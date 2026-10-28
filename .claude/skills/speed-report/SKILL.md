---
name: speed-report
description: Build the HTML report comparing ad retrieval latency against LLM response generation latency per model, from the speed_comparison benchmark runs. Use when asked to compare, summarize, or report speed comparison / retrieval vs LLM latency results.
---

# Speed comparison report

Runs live in `data/processed/speed_comparison/run_YYYYMMDD_HHMMSS/`:
- **Retrieval-only run**: contains `retrieval_speed_comparison_results.csv` (from `--retrieval-only`).
- **LLM run**: contains `speed_comparison_results.csv`, plus `speed_comparison_summary.json` (holds `llm_model`) once it completes.

The report script handles the selection rules:
- Retrieval timings come **only** from the latest retrieval-only run.
- LLM timings come from every other completed run; one row per model (run with the most queries, latest on ties).
- Runs without a summary JSON never finished and are skipped (listed in the report's notes).
- Median speedup is a per-query median (LLM time ÷ retrieval time), joined to the retrieval run on query `id`. The model table has no headroom or retrieval-first columns.

The report labels the hardware, using constants at the top of the script: `LLM_HARDWARE` (shown in the "LLM generation by model" heading, currently A100 80GB GPU) and `RETRIEVAL_HARDWARE` (the configuration line under "Retrieval stage breakdown", currently the Intel i7-8700 CPU, 15 GB RAM, no GPU). They are fixed text, not detected, so update them if the benchmark hardware changes.

## Steps

1. Generate the report (do not read `data/` directly; the script prints a compact summary):
   ```
   ./run_vllm.sh -m src.speed_comparison.scripts.build_report
   ```
   Options: `--runs-dir <dir>`, `--output <path>` (default `data/processed/speed_comparison/reports/speed_comparison_report_<newest run timestamp>.html`; rerunning updates that file in place, and a new benchmark run creates a new file).
2. Check the printed lines: every model should show `n` equal to the benchmark size and `matched == n`. Flag any skipped runs, mismatched counts, or a retrieval device that differs from what the user expects (it is in the retrieval run's summary and in the report header), and check that the hardware labels still match where the runs were done.
3. Tell the user the output path and the key numbers (retrieval p50, each model's LLM p50, tokens/s, median speedup). Offer to publish it as an Artifact if they want a shareable page.

If the user's run layout differs from these rules (e.g. they want a specific run as the retrieval source), edit `src/speed_comparison/scripts/build_report.py` rather than working around it by hand.
