import numpy as np
import pytest

from queuesight import conformal as cp
from queuesight.backtest import run_backtest
from queuesight.book import LOBData
from queuesight.features import DOWN, FLAT, UP, build_features, make_labels, multilevel_ofi
from queuesight.pipeline import ExperimentConfig, run_experiment, strategy_table
from queuesight.simulator import SimConfig, simulate


@pytest.fixture(scope="module")
def sim():
    return simulate(SimConfig(n_events=20_000, seed=3))


def _flat_book(T=50, bid=99.99, ask=100.01, tick=0.01, L=3):
    off = np.arange(L) * tick
    return LOBData(
        time=np.arange(T, dtype=float),
        ask_px=np.tile(ask + off, (T, 1)), ask_sz=np.full((T, L), 5.0),
        bid_px=np.tile(bid - off, (T, 1)), bid_sz=np.full((T, L), 5.0),
        event_type=np.zeros(T, np.int8), event_side=np.ones(T, np.int8),
        event_size=np.ones(T), tick=tick,
    )


# ---------------------------------------------------------------- simulator

def test_book_never_crosses_and_sizes_valid(sim):
    assert (sim.ask_px[:, 0] > sim.bid_px[:, 0]).all()
    assert (sim.ask_sz >= 0).all() and (sim.bid_sz >= 0).all()
    assert (sim.ask_sz[:, 0] > 0).all() and (sim.bid_sz[:, 0] > 0).all()
    assert np.all(np.diff(sim.time) > 0)


def test_simulator_is_deterministic():
    a = simulate(SimConfig(n_events=2_000, seed=11))
    b = simulate(SimConfig(n_events=2_000, seed=11))
    np.testing.assert_array_equal(a.ask_sz, b.ask_sz)
    np.testing.assert_array_equal(a.mid, b.mid)


def test_informed_regime_moves_price(sim):
    m, h = sim.mid_ticks, 200
    fut = m[h:] - m[:-h]
    reg = sim.regime[:-h]
    assert fut[reg == 1].mean() > 0 > fut[reg == -1].mean()


# ----------------------------------------------------------------- features

def test_features_have_no_lookahead(sim):
    full = build_features(sim)
    t = 15_000
    truncated = build_features(sim.slice(0, t + 1))
    np.testing.assert_allclose(full.iloc[t].to_numpy(), truncated.iloc[t].to_numpy(), equal_nan=True)


def test_ofi_sign_conventions():
    d = _flat_book(T=3)
    d.bid_sz[1, 0] = 8.0   # bid queue grows at same price -> buying pressure
    d.ask_sz[2, 0] = 9.0   # ask queue grows at same price -> selling pressure
    ofi = multilevel_ofi(d)[:, 0]
    assert ofi[1] == pytest.approx(3.0)
    assert ofi[2] == pytest.approx(-3.0 - 4.0)  # bid shrinks back (-3), ask grows (+4 against)


def test_labels_on_known_path():
    d = _flat_book(T=6)
    mids = np.array([100.0, 100.0, 100.02, 100.02, 99.98, 99.98])
    d.bid_px = d.bid_px + (mids - 100.0)[:, None]
    d.ask_px = d.ask_px + (mids - 100.0)[:, None]
    y, move = make_labels(d, horizon=2, threshold=0.5)
    assert move[0] == pytest.approx(1.0)          # mean(+0, +2) ticks
    assert y[0] == UP
    assert y[2] == DOWN                           # mean(0, -4) = -2
    assert y[-1] == -1 and y[-2] == -1            # no full horizon


# ---------------------------------------------------------------- conformal

def _exchangeable(n, seed):
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(3) * 0.8, size=n)
    y = np.array([rng.choice(3, p=p) for p in probs])
    return probs, y


@pytest.mark.parametrize("mondrian", [False, True])
def test_split_conformal_coverage(mondrian):
    alpha = 0.2
    pc, yc = _exchangeable(4_000, 0)
    pt, yt = _exchangeable(20_000, 1)
    sets = cp.SplitConformal(alpha, mondrian).fit(pc, yc).predict_sets(pt)
    m = cp.set_metrics(sets, yt)
    assert m["coverage"] >= 1 - alpha - 0.015
    if mondrian:
        for c in ("down", "flat", "up"):
            assert m[f"coverage_{c}"] >= 1 - alpha - 0.03


def test_aci_tracks_target_under_shift():
    alpha = 0.2
    pc, yc = _exchangeable(3_000, 2)
    pt, yt = _exchangeable(20_000, 3)
    yt[10_000:] = np.random.default_rng(4).integers(0, 3, 10_000)  # model goes stale
    sets, _ = cp.adaptive_conformal(pt, yt, cp.nonconformity(pc, yc), alpha, gamma=0.01, delay=50)
    assert abs(cp.set_metrics(sets, yt)["coverage"] - (1 - alpha)) < 0.02
    split = cp.SplitConformal(alpha).fit(pc, yc).predict_sets(pt)
    assert cp.set_metrics(split[10_000:], yt[10_000:])["coverage"] < 1 - alpha - 0.05


def test_gate_only_trades_directional_singletons():
    sets = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 0, 0]], bool)
    np.testing.assert_array_equal(cp.gate_signals(sets), [1, -1, 0, 0, 0])
    assert FLAT == 1


def test_quantile_edge_cases():
    s = np.sort(np.linspace(0, 1, 9))
    assert cp.conformal_quantile(s, 0.0) == np.inf
    assert cp.conformal_quantile(s, 0.05) == np.inf    # ceil(10 * .95) = 10 > n
    assert cp.conformal_quantile(s, 0.5) == pytest.approx(s[4])


# ----------------------------------------------------------------- backtest

def test_backtest_pays_the_spread_and_never_overlaps():
    d = _flat_book(T=50)
    sig = np.zeros(50, np.int8)
    sig[[0, 3, 10]] = [1, -1, -1]
    bt = run_backtest(d, np.arange(50), sig, horizon=5, latency=1, fee_ticks=0.25)
    np.testing.assert_array_equal(bt.entries, [1, 11])     # signal at t=3 is blocked
    np.testing.assert_allclose(bt.pnl_net, [-2.5, -2.5])   # 2-tick spread + 2 * 0.25 fee
    np.testing.assert_allclose(bt.pnl_mid, [0.0, 0.0])


# ------------------------------------------------------------------- pipeline

def test_end_to_end_smoke():
    cfg = ExperimentConfig(sim=SimConfig(n_events=40_000, seed=5), horizons=(10, 100),
                           horizon=100, deep=False)
    res = run_experiment(cfg)
    m = res.primary.metrics
    assert m.loc["Gradient boosting", "accuracy"] > m.loc["Class prior", "accuracy"]
    table = strategy_table(res)
    assert {"trades", "net_pnl_per_trade", "signal_rate"} <= set(table.columns)


# ------------------------------------------------------------- risk control

def _ltt_world(n, rng):
    """Conviction u ~ U(0,1); a long loses w.p. 0.6 - 0.5u. True loss rate above λ is 0.35 - 0.25λ."""
    u = rng.random(n)
    probs = np.c_[(1 - u) / 3, (1 - u) / 3, u + (1 - u) / 3]
    pnl = np.where(rng.random(n) < 0.6 - 0.5 * u, -1.0, 1.0)
    return probs, pnl


def test_ltt_guarantee_holds_at_rate_delta():
    from queuesight.risk_control import LTTGate
    rng = np.random.default_rng(0)
    target, delta, violations, traded = 0.25, 0.1, 0, 0
    for _ in range(300):
        probs, pnl = _ltt_world(400, rng)
        gate = LTTGate(target=target, delta=delta).fit(probs, pnl, -pnl, stride=1)
        if np.isfinite(gate.lam):
            traded += 1
            violations += (0.35 - 0.25 * gate.lam) > target
    assert traded > 100                       # it does find certifiable thresholds
    assert violations / 300 <= delta          # and is wrong no more often than promised


def test_ltt_abstains_when_nothing_is_certifiable():
    from queuesight.risk_control import LTTGate
    probs, pnl = _ltt_world(400, np.random.default_rng(1))
    gate = LTTGate(target=0.05).fit(probs, pnl, -pnl, stride=1)
    assert gate.lam == np.inf
    assert (gate.signals(probs) == 0).all()


def test_trade_outcomes_match_backtest_accounting():
    from queuesight.backtest import trade_outcomes
    d = _flat_book(T=50)
    long, short = trade_outcomes(d, np.array([0, 10]), horizon=5, latency=1, fee_ticks=0.25)
    np.testing.assert_allclose(long, [-2.5, -2.5])
    np.testing.assert_allclose(short, [-2.5, -2.5])
