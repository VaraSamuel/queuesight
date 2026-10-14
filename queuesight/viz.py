"""Plotly figures shared by the Streamlit app and the static HTML report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import conformal as cp
from .backtest import run_backtest
from .pipeline import Results, gated_signals

# validated categorical order (blue, orange, aqua) + recessive chrome
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SEQ = [[0, "#fcfcfb"], [0.15, "#cde2fb"], [0.4, "#86b6ef"], [0.7, "#2a78d6"], [1, "#0d366b"]]


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16, color=INK)),
        height=height, margin=dict(l=56, r=24, t=56, b=48),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=MUTED),
        legend=dict(orientation="h", y=-0.18, x=0), hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID, zeroline=False)
    return fig


def book_heatmap(res: Results, start: int | None = None, n: int = 3000) -> go.Figure:
    """Bookmap-style liquidity heatmap with mid price and the hidden regime."""
    d = res.data
    if start is None:
        start = res.split.test[0]
        if d.regime is not None:  # centre the window on a regime change so the band tells a story
            test_reg = d.regime[res.split.test]
            flips = np.flatnonzero(np.diff(test_reg) != 0)
            if len(flips):
                start = max(res.split.test[0], res.split.test[flips[0]] - n // 2)
    s = slice(start, start + n)
    tick = d.tick
    bid_t = np.round(d.bid_px[s] / tick).astype(int)
    ask_t = np.round(d.ask_px[s] / tick).astype(int)
    lo, hi = bid_t.min(), ask_t.max()
    z = np.zeros((hi - lo + 1, n))
    cols = np.arange(n)
    for lvl in range(d.n_levels):
        z[bid_t[:, lvl] - lo, cols] += d.bid_sz[s][:, lvl]
        z[ask_t[:, lvl] - lo, cols] += d.ask_sz[s][:, lvl]
    x = np.arange(start, start + n)
    has_regime = d.regime is not None
    fig = make_subplots(rows=2 if has_regime else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.85, 0.15] if has_regime else [1.0], vertical_spacing=0.04)
    fig.add_trace(go.Heatmap(x=x, y=(np.arange(lo, hi + 1)) * tick, z=np.log1p(z), colorscale=SEQ,
                             showscale=False, hovertemplate="event %{x}<br>price %{y:.2f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=d.mid[s], mode="lines", line=dict(color=ORANGE, width=2),
                             name="mid price"), 1, 1)
    if has_regime:
        fig.add_trace(go.Heatmap(x=x, y=["hidden regime"], z=d.regime[s][None, :].astype(float),
                                 zmin=-1, zmax=1, showscale=False,
                                 colorscale=[[0, ORANGE], [0.5, "#d9d8d3"], [1, BLUE]],
                                 hovertemplate="regime %{z:+.0f}<extra></extra>"), 2, 1)
    fig = _style(fig, "Order book liquidity (log size) — test period", 460)
    fig.update_layout(hovermode="closest", showlegend=False)
    fig.update_yaxes(title_text="price", row=1, col=1)
    return fig


def horizon_chart(table: pd.DataFrame) -> go.Figure:
    """Accuracy illusion: accuracy rises at short horizons, tradable edge does not."""
    h = [f"{k} events" for k in table.index]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12,
                        subplot_titles=("Classification quality", "Net PnL per trade (ticks, after spread)"))
    fig.add_trace(go.Bar(x=h, y=table["accuracy"], name="accuracy", marker_color=BLUE), 1, 1)
    fig.add_trace(go.Bar(x=h, y=table["prior_accuracy"], name="majority-class baseline",
                         marker_color="#b5b4ad"), 1, 1)
    fig.add_trace(go.Bar(x=h, y=table["directional_precision"], name="directional precision",
                         marker_color=AQUA), 1, 1)
    fig.add_trace(go.Bar(x=h, y=table["net_pnl_per_trade"], name="net PnL / trade",
                         marker_color=ORANGE, showlegend=False,
                         text=[f"{v:+.2f}" for v in table["net_pnl_per_trade"]],
                         textposition="outside"), 1, 2)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1), row=1, col=2)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_traces(marker_line_width=0)
    return _style(fig, "The accuracy illusion: horizon vs tradability", 400)


def coverage_chart(report: pd.DataFrame, alpha: float) -> go.Figure:
    cols = ["coverage", "coverage_down", "coverage_flat", "coverage_up"]
    labels = ["overall", "down", "flat", "up"]
    fig = go.Figure()
    for (name, row), color in zip(report.iterrows(), (BLUE, ORANGE, AQUA)):
        fig.add_trace(go.Bar(x=labels, y=row[cols].to_numpy(), name=name, marker_color=color))
    fig.add_hline(y=1 - alpha, line=dict(color=INK, width=1),
                  annotation_text=f"target 1-α = {1 - alpha:.2f}", annotation_position="top left")
    fig.update_layout(barmode="group", yaxis_range=[0, 1], bargap=0.25, bargroupgap=0.08)
    return _style(fig, "Empirical coverage on unseen test data", 380)


def rolling_coverage_chart(res: Results, model: str, alpha: float, window: int = 3000) -> go.Figure:
    run, s = res.primary, res.split
    pc, pt, yc, yt = run.probs_cal[model], run.probs_test[model], run.y[s.cal], run.y[s.test]
    split_sets = cp.SplitConformal(alpha).fit(pc, yc).predict_sets(pt)
    aci_sets, alpha_path = cp.adaptive_conformal(pt, yt, cp.nonconformity(pc, yc), alpha,
                                                 res.cfg.aci_gamma, delay=run.horizon)
    step = max(1, len(yt) // 1500)  # thin the plotted points; rolling stat is computed on all
    roll = lambda sets: pd.Series(sets[np.arange(len(yt)), yt].astype(float)).rolling(window).mean()[::step]  # noqa: E731
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.test[::step], y=roll(split_sets), name="split conformal",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=s.test[::step], y=roll(aci_sets), name="adaptive (delayed-label ACI)",
                             line=dict(color=ORANGE, width=2)))
    fig.add_hline(y=1 - alpha, line=dict(color=INK, width=1))
    fig.update_yaxes(title_text=f"rolling coverage ({window} events)")
    fig.update_xaxes(title_text="event")
    return _style(fig, "Coverage through regime shifts", 360)


def frontier_chart(fr: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for name, color in [("Conformal gate", BLUE), ("Confidence gate (matched)", ORANGE)]:
        sub = fr[fr["strategy"] == name].sort_values("signal_rate")
        fig.add_trace(go.Scatter(
            x=sub["signal_rate"], y=sub["net_pnl_per_trade"], mode="lines+markers", name=name,
            line=dict(color=color, width=2), marker=dict(size=9, line=dict(color=SURFACE, width=2)),
            customdata=np.c_[sub["alpha"], sub["trades"], sub["hit_rate"]],
            hovertemplate="α=%{customdata[0]:.2f}<br>signal rate %{x:.0%}<br>"
                          "net %{y:+.2f} ticks/trade<br>%{customdata[1]} trades, "
                          "hit %{customdata[2]:.0%}<extra>" + name + "</extra>"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="fraction of events with a trade signal", tickformat=".0%")
    fig.update_yaxes(title_text="net PnL per trade (ticks)")
    fig.update_layout(hovermode="closest")
    return _style(fig, "Selectivity frontier: sweep α, compare at equal trade rates", 400)


def equity_chart(res: Results, model: str, alpha: float, latency: int, fee: float) -> go.Figure:
    run, s, h = res.primary, res.split, res.cfg.horizon
    pc, pt = run.probs_cal[model], run.probs_test[model]
    cal_rate = (cp.gate_signals(cp.SplitConformal(alpha, True).fit(pc, run.y[s.cal]).predict_sets(pc)) != 0).mean()
    strategies = [
        ("Always trade argmax", cp.argmax_signals(pt), MUTED),
        ("Confidence gate (matched)", cp.threshold_signals(pt, cp.matched_tau(pc, cal_rate)), ORANGE),
        (f"Conformal gate α={alpha:.2f}", gated_signals(res, model, alpha, "mondrian"), BLUE),
    ]
    fig = go.Figure()
    for name, sig, color in strategies:
        bt = run_backtest(res.data, s.test, sig, h, latency, fee)
        fig.add_trace(go.Scatter(x=bt.entries, y=np.cumsum(bt.pnl_net), name=name, mode="lines",
                                 line=dict(color=color, width=2)))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_yaxes(title_text="cumulative net PnL (ticks)")
    fig.update_xaxes(title_text="event")
    return _style(fig, "Equity curves — crossing the spread, non-overlapping trades", 380)


def latency_chart(lc: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=lc["latency_events"], y=lc["net_pnl_per_trade"], mode="lines+markers",
                               line=dict(color=BLUE, width=2), marker=dict(size=9),
                               name="net PnL / trade"))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    fig.update_xaxes(title_text="latency (events between signal and fill)")
    fig.update_yaxes(title_text="net PnL per trade (ticks)")
    return _style(fig, "Alpha decay with latency", 340)


def importance_chart(imp: pd.Series, top: int = 12) -> go.Figure:
    imp = imp.tail(top)
    fig = go.Figure(go.Bar(x=imp.to_numpy(), y=imp.index, orientation="h", marker_color=BLUE,
                           hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title_text="log-loss increase when shuffled")
    fig.update_layout(hovermode="closest")
    return _style(fig, "What the model uses (permutation importance)", 400)


def regime_chart(res: Results, model: str, n: int = 20_000) -> go.Figure | None:
    d = res.data
    if d.regime is None:
        return None
    idx = res.split.test[:n]
    p = res.primary.probs_test[model][: len(idx)]
    step = max(1, len(idx) // 2000)
    score = pd.Series(p[:, 2] - p[:, 0]).rolling(200, min_periods=1).mean().to_numpy()[::step]
    idx = idx[::step]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=d.regime[idx], name="hidden informed regime",
                             line=dict(color="#b5b4ad", width=2, shape="hv")))
    fig.add_trace(go.Scatter(x=idx, y=score / max(np.abs(score).max(), 1e-9), name="model P(up) − P(down), scaled",
                             line=dict(color=BLUE, width=2)))
    fig.update_yaxes(title_text="bearish ← → bullish", range=[-1.15, 1.15])
    fig.update_xaxes(title_text="event")
    return _style(fig, "The model infers a regime it never sees", 360)
