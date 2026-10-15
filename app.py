"""QueueSight dashboard.   streamlit run app.py"""

from __future__ import annotations

import streamlit as st

from queuesight import viz
from queuesight.pipeline import (
    ExperimentConfig,
    alpha_frontier,
    conformal_report,
    horizon_table,
    latency_curve,
    ltt_report,
    regime_recovery,
    run_experiment,
    strategy_table,
)
from queuesight.simulator import SimConfig

st.set_page_config(page_title="QueueSight", layout="wide")


@st.cache_resource(show_spinner=False)
def experiment(events: int, seed: int, horizon: int, informed: float, deep: bool):
    sim = SimConfig(n_events=events, seed=seed, informed_strength=informed)
    cfg = ExperimentConfig(sim=sim, horizons=(10, 50, horizon), horizon=horizon, deep=deep)
    return run_experiment(cfg)


with st.sidebar:
    st.header("Market")
    events = st.select_slider("Simulated events", [200_000, 300_000, 600_000, 1_000_000], 600_000)
    seed = st.number_input("Seed", 0, 10_000, 7)
    informed = st.slider("Informed trader strength", 0.0, 0.4, 0.18, 0.02,
                         help="How hard the hidden regime tilts aggressive order flow. 0 = no edge exists.")
    horizon = st.select_slider("Forecast horizon (events)", [50, 100, 200, 400], 200)
    deep = st.checkbox("Train DeepLOB-lite (PyTorch, ~1 min)", value=False)
    st.header("Trading")
    alpha = st.slider("Conformal α (miscoverage budget)", 0.05, 0.5, 0.25, 0.05,
                      help="Lower α = larger prediction sets = fewer, higher-conviction trades.")
    risk_target = st.slider("Max losing-trade rate (LTT)", 0.25, 0.50, 0.40, 0.01,
                            help="Learn-then-Test only trades thresholds it can certify at this "
                                 "loss rate with 90% confidence. Too strict and it won't trade.")
    latency = st.slider("Latency (events)", 0, 40, 1)
    fee = st.slider("Fee per side (ticks)", 0.0, 1.0, 0.0, 0.05)

st.title("QueueSight")
st.caption("Limit order book forecasting with distribution-free trade gating. "
           "Every number below is after crossing the spread.")

with st.spinner("Simulating the book and training models…"):
    res = experiment(events, int(seed), horizon, informed, deep)

models = list(res.primary.probs_test)
model = st.selectbox("Forecaster", models, index=models.index(res.best_model),
                     help="Default = lowest log-loss on the calibration window.")
strat = strategy_table(res, model, alpha, latency, fee, risk_target)
ltt_row = strat.loc[[i for i in strat.index if i.startswith("Risk-controlled")][0]]
base_row = strat.loc["Always trade argmax"]
ltt = ltt_report(res, model, risk_target, latency, fee)
reg = regime_recovery(res, model)

c1, c2, c3, c4 = st.columns(4)
acc = res.primary.metrics.loc[model, "accuracy"]
c1.metric("Test accuracy", f"{acc:.1%}", f"{acc - res.primary.metrics.loc['Class prior', 'accuracy']:+.1%} vs prior")
if ltt_row["trades"] > 0:
    c2.metric("LTT net / trade", f"{ltt_row['net_pnl_per_trade']:+.2f} ticks",
              f"{ltt_row['net_pnl_per_trade'] - base_row['net_pnl_per_trade']:+.2f} vs argmax")
    c3.metric("LTT losing trades (test)", f"{ltt['test_loss_rate']:.0%}", f"target ≤ {risk_target:.0%}",
              delta_color="off")
else:
    c2.metric("LTT net / trade", "abstained")
    c3.metric("LTT losing trades (test)", "n/a", "nothing certifiable at this target", delta_color="off")
c4.metric("Hidden-regime corr.", f"{reg['corr']:.2f}" if reg else "n/a")

tabs = st.tabs(["Order book", "Accuracy vs edge", "Conformal sets", "Risk control", "Trading", "Features"])
with tabs[0]:
    st.plotly_chart(viz.book_heatmap(res), width="stretch")
    fig = viz.regime_chart(res, model)
    if fig is not None:
        st.plotly_chart(fig, width="stretch")
with tabs[1]:
    st.markdown("Short horizons look great on accuracy because the mid rarely moves, so the model "
                "learns to say *flat*. The edge that survives the spread lives further out.")
    hz = horizon_table(res)
    st.plotly_chart(viz.horizon_chart(hz), width="stretch")
    st.dataframe(hz.round(3), width="stretch")
    for h, r in res.runs.items():
        st.caption(f"h = {h} events · flat band ±{r.threshold:.2f} ticks")
        st.dataframe(r.metrics.round(3), width="stretch")
with tabs[2]:
    cov = conformal_report(res, model, alpha)
    st.plotly_chart(viz.coverage_chart(cov, alpha), width="stretch")
    st.dataframe(cov.round(3), width="stretch")
    st.plotly_chart(viz.rolling_coverage_chart(res, model, alpha), width="stretch")
with tabs[3]:
    st.markdown(f"Each candidate threshold is a binomial test of *H0: losing-trade rate > {risk_target:.0%}* "
                "on calibration outcomes thinned to one per horizon, Bonferroni-corrected at δ = 0.10. "
                "The gate uses the loosest threshold that passes.")
    st.plotly_chart(viz.ltt_chart(ltt), width="stretch")
    st.dataframe(ltt["trail"].round(3), width="stretch")
with tabs[4]:
    st.dataframe(strat.round(3), width="stretch")
    st.plotly_chart(viz.equity_chart(res, model, alpha, latency, fee), width="stretch")
    st.plotly_chart(viz.frontier_chart(alpha_frontier(res, model, latency=latency, fee_ticks=fee)), width="stretch")
    st.plotly_chart(viz.latency_chart(latency_curve(res, model, alpha, fee_ticks=fee)), width="stretch")
with tabs[5]:
    if res.importance is not None:
        st.plotly_chart(viz.importance_chart(res.importance), width="stretch")
