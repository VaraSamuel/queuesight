"""CLI: run the experiment and write a standalone HTML report + markdown summary.

    python -m queuesight.report --events 300000 --horizon 200 --out results
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from . import viz
from .pipeline import (ExperimentConfig, alpha_frontier, conformal_report, horizon_table,
                       latency_curve, regime_recovery, run_experiment, strategy_table)
from .simulator import SimConfig

CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e0}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,system-ui,sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px}
p.lede{color:var(--muted);margin:0 0 24px;max-width:760px}
table{border-collapse:collapse;font-size:13px;margin:8px 0 16px;width:100%;overflow-x:auto;display:block}
th,td{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
"""


def _table(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_html(border=0, na_rep="–")


def _markdown(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.round(digits).reset_index()
    rows = [list(map(str, df.columns))] + [[str(v) for v in r] for r in df.itertuples(index=False)]
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    lines.insert(1, "|" + "---|" * len(df.columns))
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=300_000)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--latency", type=int, default=1)
    ap.add_argument("--fee", type=float, default=0.0, help="fee per side, in ticks")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--deep", action="store_true", help="also train the PyTorch DeepLOB-lite model")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)

    cfg = ExperimentConfig(sim=SimConfig(n_events=a.events, seed=a.seed), horizon=a.horizon,
                           alpha=a.alpha, latency=a.latency, fee_ticks=a.fee, deep=a.deep)
    t0 = time.time()
    res = run_experiment(cfg, progress=lambda m: print(f"[{time.time() - t0:6.1f}s] {m}"))
    model = res.best_model

    hz = horizon_table(res)
    cov = conformal_report(res, model)
    strat = strategy_table(res, model)
    fr = alpha_frontier(res, model)
    lat = latency_curve(res, model)
    reg = regime_recovery(res, model)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    figs = [
        viz.book_heatmap(res), viz.horizon_chart(hz), viz.regime_chart(res, model),
        viz.coverage_chart(cov, cfg.alpha), viz.rolling_coverage_chart(res, model, cfg.alpha),
        viz.frontier_chart(fr), viz.equity_chart(res, model, cfg.alpha, cfg.latency, cfg.fee_ticks),
        viz.latency_chart(lat), viz.importance_chart(res.importance),
    ]
    html_figs = [f.to_html(full_html=False, include_plotlyjs="cdn" if i == 0 else False)
                 for i, f in enumerate(x for x in figs if x is not None)]

    models_html = "".join(f"<h3>horizon = {h} events (flat band ±{r.threshold:.2f} ticks)</h3>"
                          + _table(r.metrics) for h, r in res.runs.items())
    reg_line = (f"<p>Correlation between P(up)−P(down) and the hidden regime: "
                f"<b>{reg['corr']:.2f}</b></p>" if reg else "")
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>QueueSight report</title>
<style>{CSS}</style></head><body><main>
<h1>QueueSight</h1>
<p class="lede">Conformal-gated limit order book forecasting. {a.events:,} simulated events,
primary horizon {cfg.horizon} events, α = {cfg.alpha}, latency {cfg.latency} event(s),
fee {cfg.fee_ticks} ticks/side. Selected model (lowest calibration log-loss): <b>{model}</b>.</p>
{html_figs[0]}
<h2>1 · Accuracy is not edge</h2>{_table(hz)}{html_figs[1]}
<h2>2 · Models</h2>{models_html}{reg_line}{html_figs[2]}
<h2>3 · Conformal coverage</h2>{_table(cov)}{html_figs[3]}{html_figs[4]}
<h2>4 · Trading the sets</h2>{_table(strat)}{html_figs[5]}{html_figs[6]}{html_figs[7]}
<h2>5 · Features</h2>{html_figs[8]}
</main></body></html>"""
    (out / "report.html").write_text(html)

    summary = {
        "config": {"events": a.events, "horizon": cfg.horizon, "alpha": cfg.alpha,
                   "latency": cfg.latency, "fee_ticks": cfg.fee_ticks, "seed": a.seed},
        "best_model": model, "regime_recovery": reg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    md = ["## Horizon vs tradability", _markdown(hz), "## Conformal coverage", _markdown(cov),
          "## Strategies", _markdown(strat)]
    (out / "summary.md").write_text("\n\n".join(md) + "\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(hz.round(3), cov.round(3), strat.round(3), sep="\n\n")
    print(f"\nWrote {out / 'report.html'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
