@AGENTS.md

## Commands
- Run via `./run_vllm.sh -m <module>` (unsets `LD_LIBRARY_PATH`, uses `.lmarena-env`), from repo root. Example: `./run_vllm.sh -m src.ad_evaluation.scripts.score_placements --limit 100`
- Tests (unittest, no pytest/lint/typecheck configured): `./run_vllm.sh -m unittest discover -s tests -t .`
- Env vars in `.env` (see `.env.example`: `HF_TOKEN`, `GEMINI_API_KEY`).

## Docs
- Data flow: `docs/architecture.md` (read when a change spans modules)
- Coding standards: `.agents/rules/python.md` (pragmatic SOLID; no abstractions for their own sake)
- `src/speed_comparison/AGENTS.md` has module-specific notes

## Gotchas
- Never read or dump `data/`, `.lmarena-env/`, or notebook outputs in `analysis/`; they are large.
- `src/ad_generation/llm/__init__.py` and `src/common/shared_llm.py` have uncommitted user edits; don't revert them.
- Unit tests must not hit the network or GPU.

- Project map (module responsibilities, data flow, fragile spots, "where to look"): `docs/PROJECT_MAP.md` (read on demand).
