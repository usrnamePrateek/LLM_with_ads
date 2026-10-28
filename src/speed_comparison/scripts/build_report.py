"""Builds an HTML report comparing ad retrieval latency against LLM generation latency per model.

Retrieval timings come from the latest retrieval-only run; LLM timings come from every other completed
run (one row per LLM model, keeping the run with the most queries, latest on ties). Runs without a
summary JSON never finished and are listed as skipped.
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.speed_comparison.config import (
    DEFAULT_OUTPUT_CSV,
    DEFAULT_RUNS_DIR,
    DEFAULT_SPEED_COMPARISON_DIR,
    DEFAULT_SUMMARY_JSON,
)

RETRIEVAL_CSV_NAME = f"retrieval_{DEFAULT_OUTPUT_CSV.name}"
LLM_HARDWARE = "A100 80GB GPU"
RETRIEVAL_HARDWARE = "Intel Core i7-8700 (6 cores / 12 threads, 3.2 GHz base, 4.6 GHz boost) · 15 GB RAM · no GPU"
DEFAULT_REPORTS_DIR = DEFAULT_SPEED_COMPARISON_DIR / "reports"


def _load_summary(run_dir: Path) -> dict:
    path = run_dir / DEFAULT_SUMMARY_JSON.name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _pct(values: pd.Series, q: float) -> float:
    return float(np.percentile(values.to_numpy(dtype=float), q))


def find_runs(base_dir: Path) -> tuple[Path, list[Path], list[Path]]:
    """Returns (latest retrieval-only run, completed LLM runs, incomplete LLM runs), sorted by dir name."""
    runs = sorted(p for p in base_dir.glob("run_*") if p.is_dir())
    retrieval_runs = [p for p in runs if (p / RETRIEVAL_CSV_NAME).exists()]
    if not retrieval_runs:
        raise FileNotFoundError(f"no retrieval-only run ({RETRIEVAL_CSV_NAME}) under {base_dir}")
    llm_runs = [p for p in runs if (p / DEFAULT_OUTPUT_CSV.name).exists()]
    completed = [p for p in llm_runs if (p / DEFAULT_SUMMARY_JSON.name).exists()]
    incomplete = [p for p in llm_runs if p not in completed]
    return retrieval_runs[-1], completed, incomplete


def build_model_rows(retrieval: pd.DataFrame, llm_runs: list[Path]) -> list[dict]:
    """One row per LLM model, with per-query speedups joined against the retrieval run on query id."""
    retrieval_ms = retrieval.set_index("id")["total_retrieval_ms"]
    by_model: dict[str, dict] = {}
    for run_dir in llm_runs:
        frame = pd.read_csv(run_dir / DEFAULT_OUTPUT_CSV.name, dtype={"id": str})
        frame = frame.drop_duplicates("id", keep="last")
        if frame.empty:
            continue
        summary = _load_summary(run_dir)
        model = summary["llm_model"]
        joined = frame[frame["id"].isin(retrieval_ms.index)].copy()
        joined["r_ms"] = joined["id"].map(retrieval_ms)
        ratio = joined["llm_time_ms"] / joined["r_ms"]
        llm = frame["llm_time_ms"]
        row = {
            "model": model,
            "run": run_dir.name,
            "n": len(frame),
            "matched": len(joined),
            "llm_mean": float(llm.mean()),
            "llm_p50": _pct(llm, 50),
            "llm_p90": _pct(llm, 90),
            "llm_p95": _pct(llm, 95),
            "llm_p99": _pct(llm, 99),
            "tok_s": float(frame["tokens_per_sec"].mean()),
            "out_tok": float(frame["output_tokens"].mean()),
            "speedup_median": float(ratio.median()) if len(joined) else float("nan"),
            "wall_clock_sec": float(summary.get("total_llm_wall_clock_sec") or 0.0),
        }
        previous = by_model.get(model)
        if previous is None or row["n"] >= previous["n"]:
            by_model[model] = row
    return sorted(by_model.values(), key=lambda r: r["llm_p50"])


def _fmt_ms(ms: float) -> str:
    if np.isnan(ms):
        return "–"
    return f"{ms / 1000:,.2f} s" if ms >= 1000 else f"{ms:,.1f} ms"


def render_html(retrieval_run: Path, retrieval: pd.DataFrame, rows: list[dict], skipped: list[Path]) -> str:
    r_summary = _load_summary(retrieval_run)
    r_total = retrieval["total_retrieval_ms"]
    r_p50 = _pct(r_total, 50)
    stage_rows = "".join(
        f"<tr><th scope='row'>{label}</th>"
        + "".join(f"<td>{_fmt_ms(v)}</td>" for v in (
            float(retrieval[col].mean()), _pct(retrieval[col], 50), _pct(retrieval[col], 90),
            _pct(retrieval[col], 95), _pct(retrieval[col], 99),
        ))
        + "</tr>"
        for label, col in (("Total retrieval", "total_retrieval_ms"), ("Query embedding", "embed_time_ms"), ("FAISS search", "search_time_ms"))
    )

    max_p50 = max((r["llm_p50"] for r in rows), default=1.0)
    model_rows = []
    for r in rows:
        bar = 100 * r["llm_p50"] / max_p50 if max_p50 else 0
        retrieval_mark = min(100.0, 100 * r_p50 / max_p50) if max_p50 else 0
        matched_note = "" if r["matched"] == r["n"] else f"<span class='warn' title='queries matched to retrieval run'> · {r['matched']:,} matched</span>"
        model_rows.append(
            "<tr>"
            f"<th scope='row'><div class='model'>{html.escape(r['model'])}</div><div class='sub'>{html.escape(r['run'])} · n={r['n']:,}{matched_note}</div></th>"
            f"<td class='barcell'><div class='bar'><span style='width:{bar:.2f}%'></span><i style='left:{retrieval_mark:.2f}%' title='retrieval p50'></i></div></td>"
            f"<td>{_fmt_ms(r['llm_p50'])}</td><td>{_fmt_ms(r['llm_mean'])}</td><td>{_fmt_ms(r['llm_p90'])}</td>"
            f"<td>{_fmt_ms(r['llm_p95'])}</td><td>{_fmt_ms(r['llm_p99'])}</td>"
            f"<td>{r['tok_s']:,.1f}</td><td>{r['out_tok']:,.0f}</td>"
            f"<td class='hl'>{r['speedup_median']:,.0f}×</td>"
            "</tr>"
        )

    fastest = rows[0] if rows else None
    min_speedup = min((r["speedup_median"] for r in rows), default=float("nan"))
    throughput = len(retrieval) / r_summary["total_retrieval_wall_clock_sec"] if r_summary.get("total_retrieval_wall_clock_sec") else float("nan")
    device = html.escape(str(r_summary.get("retrieval_device", "?")))
    embed_model = html.escape(str(r_summary.get("embedding_model", "BAAI/bge-m3")))
    skipped_note = (
        "<li>Skipped incomplete runs (no summary JSON, model unknown): "
        + ", ".join(f"<code>{html.escape(p.name)}</code>" for p in skipped)
        + ".</li>"
    ) if skipped else ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retrieval vs LLM Latency</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--text:#1d1d1f;--muted:#6b6b70;--line:#e4e4e0;--accent:#2a6fdb;--accent-soft:#d6e4fa;--mark:#d9480f;--warn:#b35c00}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#141416;--card:#1d1d20;--text:#ececef;--muted:#9a9aa2;--line:#303035;--accent:#6ea2f5;--accent-soft:#24344f;--mark:#ff8a4c;--warn:#f0a75a}}}}
:root[data-theme="dark"]{{--bg:#141416;--card:#1d1d20;--text:#ececef;--muted:#9a9aa2;--line:#303035;--accent:#6ea2f5;--accent-soft:#24344f;--mark:#ff8a4c;--warn:#f0a75a}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:40px 16px 64px}}
h1{{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}} h2{{font-size:17px;margin:40px 0 12px}}
.hw{{color:var(--muted);font-weight:400}} .config{{color:var(--muted);font-size:13px;margin:0 2px 10px}}
.lede{{color:var(--muted);margin:0 0 28px;max-width:760px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px}}
.tile .k{{color:var(--muted);font-size:13px}} .tile .v{{font-size:26px;font-weight:600;font-variant-numeric:tabular-nums;margin-top:2px}}
.tile .s{{color:var(--muted);font-size:12px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
thead th{{font-size:12px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.03em;background:var(--card);position:sticky;top:0}}
tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}
tbody th{{text-align:left;font-weight:500}}
.model{{font-weight:600}} .sub{{color:var(--muted);font-size:12px}} .warn{{color:var(--warn)}}
.hl{{font-weight:600;color:var(--accent)}}
.barcell{{min-width:160px;width:18%}}
.bar{{position:relative;height:10px;background:var(--line);border-radius:5px}}
.bar span{{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:5px}}
.bar i{{position:absolute;top:-4px;bottom:-4px;width:2px;background:var(--mark)}}
.legend{{color:var(--muted);font-size:13px;margin:10px 2px 0}} .legend b{{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:4px}}
.notes{{color:var(--muted);font-size:13px;margin-top:28px}} .notes li{{margin:4px 0}}
</style></head><body><main>
<h1>Ad retrieval vs LLM response latency</h1>
<p class="lede">Ad retrieval ({embed_model} on {device} + FAISS) from <code>{html.escape(retrieval_run.name)}</code>, compared against LLM response generation for each model. Benchmark: {len(retrieval):,} queries, batch size 1. Generated {datetime.now():%Y-%m-%d %H:%M}.</p>

<div class="tiles">
<div class="tile"><div class="k">Retrieval p50</div><div class="v">{_fmt_ms(r_p50)}</div><div class="s">p95 {_fmt_ms(_pct(r_total, 95))} · p99 {_fmt_ms(_pct(r_total, 99))}</div></div>
<div class="tile"><div class="k">Retrieval throughput</div><div class="v">{throughput:,.1f}/s</div><div class="s">queries per second, sequential</div></div>
<div class="tile"><div class="k">Fastest LLM p50</div><div class="v">{_fmt_ms(fastest["llm_p50"]) if fastest else "–"}</div><div class="s">{html.escape(fastest["model"]) if fastest else ""}</div></div>
<div class="tile"><div class="k">Smallest median speedup</div><div class="v">{min_speedup:,.0f}×</div><div class="s">LLM time ÷ retrieval time, per query</div></div>
</div>

<h2>LLM generation by model <span class="hw">({LLM_HARDWARE})</span></h2>
<div class="card"><table>
<thead><tr><th style="text-align:left">Model</th><th style="text-align:left">LLM p50</th><th>p50</th><th>Mean</th><th>p90</th><th>p95</th><th>p99</th><th>Tokens/s</th><th>Out tokens</th><th>Median speedup</th></tr></thead>
<tbody>{"".join(model_rows)}</tbody></table></div>
<p class="legend"><b style="background:var(--accent)"></b>LLM p50 latency (scaled to slowest model) &nbsp; <b style="background:var(--mark);width:3px"></b>retrieval p50</p>

<h2>Retrieval stage breakdown</h2>
<p class="config"><b>Configuration:</b> {embed_model} on {device} · FAISS (CPU) · {html.escape(RETRIEVAL_HARDWARE)}</p>
<div class="card"><table>
<thead><tr><th style="text-align:left">Stage</th><th>Mean</th><th>p50</th><th>p90</th><th>p95</th><th>p99</th></tr></thead>
<tbody>{stage_rows}</tbody></table></div>

<ul class="notes">
<li><b>Median speedup</b> is the per-query median of LLM latency ÷ retrieval latency, joined on query id.</li>
<li>Retrieval timings come only from the latest retrieval-only run; LLM timings from earlier runs (the run with the most queries per model).</li>
{skipped_note}
</ul>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Report path (default: reports/speed_comparison_report_<newest run timestamp>.html next to runs/).",
    )
    args = parser.parse_args()

    retrieval_run, llm_runs, skipped = find_runs(args.runs_dir)
    retrieval = pd.read_csv(retrieval_run / RETRIEVAL_CSV_NAME, dtype={"id": str}).drop_duplicates("id", keep="last")
    rows = build_model_rows(retrieval, llm_runs)

    print(f"Retrieval run: {retrieval_run.name} ({len(retrieval):,} queries, p50 {_pct(retrieval['total_retrieval_ms'], 50):.2f} ms)")
    for r in rows:
        print(
            f"  {r['model']:<45} {r['run']:<22} n={r['n']:<5} matched={r['matched']:<5} "
            f"p50={r['llm_p50']:>9.1f}ms tok/s={r['tok_s']:>6.1f} speedup={r['speedup_median']:>7.1f}x"
        )
    for run_dir in skipped:
        print(f"  skipped incomplete run (no summary JSON): {run_dir.name}")
    # Named after the newest run included, so reruns update the report and only a new run creates a new one.
    newest_run = max(p.name for p in [retrieval_run, *llm_runs]).removeprefix("run_")
    output = args.output or DEFAULT_REPORTS_DIR / f"speed_comparison_report_{newest_run}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(retrieval_run, retrieval, rows, skipped), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
