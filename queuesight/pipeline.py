"""End-to-end experiment: simulate -> features -> models -> conformal -> backtest."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, log_loss

from . import conformal as cp
from .backtest import run_backtest
from .book import LOBData
from .features import balanced_threshold, build_features, make_labels
from .models import DeepLOBLite, boosted, logistic
from .simulator import SimConfig, simulate

WARMUP = 500


@dataclass
class ExperimentConfig:
    sim: SimConfig = field(default_factory=lambda: SimConfig(n_events=300_000))
    horizons: tuple[int, ...] = (10, 50, 200)
    horizon: int = 200           # primary horizon for conformal + trading
    alpha: float = 0.25
    aci_gamma: float = 0.005
    latency: int = 1             # events between decision and fill
    fee_ticks: float = 0.0
    train_stride: int = 2        # subsample training rows (they are autocorrelated)
    deep: bool = True
    seed: int = 0


@dataclass
class Split:
    train: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def chrono_split(T: int, max_h: int, embargo: int) -> Split:
    """60/20/20 chronological split with an embargo so labels never straddle a boundary."""
    usable = np.arange(WARMUP, T - max_h - 50)
    n = len(usable)
    a, b = int(0.6 * n), int(0.8 * n)
    return Split(usable[: a - embargo], usable[a: b - embargo], usable[b:])


@dataclass
class HorizonRun:
    horizon: int
    threshold: float
    y: np.ndarray
    probs_cal: dict[str, np.ndarray]
    probs_test: dict[str, np.ndarray]
    metrics: pd.DataFrame


@dataclass
class Results:
    cfg: ExperimentConfig
    data: LOBData
    features: pd.DataFrame
    split: Split
    runs: dict[int, HorizonRun]
    best_model: str
    importance: pd.Series | None = None

    @property
    def primary(self) -> HorizonRun:
        return self.runs[self.cfg.horizon]


def _metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    directional = pred != 1
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "log_loss": log_loss(y, probs, labels=[0, 1, 2]),
        "directional_precision": float((pred[directional] == y[directional]).mean())
        if directional.any() else np.nan,
    }


def run_experiment(cfg: ExperimentConfig | None = None, data: LOBData | None = None,
                   progress=lambda msg: None) -> Results:
    cfg = cfg or ExperimentConfig()
    horizons = tuple(sorted(set(cfg.horizons) | {cfg.horizon}))
    progress("Simulating order book")
    d = data if data is not None else simulate(cfg.sim)
    progress("Building order-flow features")
    feats = build_features(d)
    split = chrono_split(len(d), max(horizons), embargo=max(horizons))
    tr = split.train[:: cfg.train_stride]

    runs: dict[int, HorizonRun] = {}
    importance = None
    for h in horizons:
        _, move = make_labels(d, h, 0.0)
        thr = balanced_threshold(move[split.train])
        y, _ = make_labels(d, h, thr)
        models = [logistic(feats), boosted(feats, cfg.seed)]
        if cfg.deep and h == cfg.horizon:
            models.append(DeepLOBLite(d, seed=cfg.seed))
        probs_cal, probs_test, rows = {}, {}, []
        for m in models:
            progress(f"h={h}: training {m.name}")
            m.fit(tr, y[tr])
            probs_cal[m.name] = m.predict_proba(split.cal)
            probs_test[m.name] = m.predict_proba(split.test)
            rows.append({"model": m.name, **_metrics(probs_test[m.name], y[split.test])})
            if h == cfg.horizon and m.name == "Gradient boosting":
                progress("Permutation importance")
                sub = split.test[:: max(1, len(split.test) // 8000)]
                pi = permutation_importance(m.est, m.X[sub], y[sub], n_repeats=3,
                                            random_state=cfg.seed, scoring="neg_log_loss")
                importance = pd.Series(pi.importances_mean, index=m.feature_names).sort_values()
        prior = np.bincount(y[tr], minlength=3) / len(tr)
        rows.append({"model": "Class prior", **_metrics(np.tile(prior, (len(split.test), 1)), y[split.test])})
        runs[h] = HorizonRun(h, thr, y, probs_cal, probs_test, pd.DataFrame(rows).set_index("model"))

    prim = runs[cfg.horizon]
    y_cal = prim.y[split.cal]
    best = min(prim.probs_cal, key=lambda k: log_loss(y_cal, prim.probs_cal[k], labels=[0, 1, 2]))
    progress("Done")
    return Results(cfg, d, feats, split, runs, best, importance)


# ---------------------------------------------------------------- analyses ---

def conformal_report(res: Results, model: str | None = None, alpha: float | None = None) -> pd.DataFrame:
    """Coverage of split / Mondrian / adaptive conformal on the test period."""
    model = model or res.best_model
    alpha = res.cfg.alpha if alpha is None else alpha
    run, s = res.primary, res.split
    pc, pt = run.probs_cal[model], run.probs_test[model]
    yc, yt = run.y[s.cal], run.y[s.test]
    rows = {}
    for name, mondrian in [("Split conformal", False), ("Mondrian (class-conditional)", True)]:
        sets = cp.SplitConformal(alpha, mondrian).fit(pc, yc).predict_sets(pt)
        rows[name] = cp.set_metrics(sets, yt)
    aci_sets, _ = cp.adaptive_conformal(pt, yt, cp.nonconformity(pc, yc), alpha,
                                        res.cfg.aci_gamma, delay=run.horizon)
    rows["Adaptive (ACI, delayed labels)"] = cp.set_metrics(aci_sets, yt)
    return pd.DataFrame(rows).T


def gated_signals(res: Results, model: str, alpha: float, method: str = "split") -> np.ndarray:
    run, s = res.primary, res.split
    pc, pt = run.probs_cal[model], run.probs_test[model]
    yc, yt = run.y[s.cal], run.y[s.test]
    if method == "aci":
        sets, _ = cp.adaptive_conformal(pt, yt, cp.nonconformity(pc, yc), alpha,
                                        res.cfg.aci_gamma, delay=run.horizon)
    else:
        sets = cp.SplitConformal(alpha, mondrian=(method == "mondrian")).fit(pc, yc).predict_sets(pt)
    return cp.gate_signals(sets)


def strategy_table(res: Results, model: str | None = None, alpha: float | None = None,
                   latency: int | None = None, fee_ticks: float | None = None) -> pd.DataFrame:
    """Argmax vs confidence threshold vs conformal gate, all execution-aware."""
    model = model or res.best_model
    alpha = res.cfg.alpha if alpha is None else alpha
    lat = res.cfg.latency if latency is None else latency
    fee = res.cfg.fee_ticks if fee_ticks is None else fee_ticks
    run, s, h = res.primary, res.split, res.cfg.horizon
    pc, pt = run.probs_cal[model], run.probs_test[model]

    conf_sig = gated_signals(res, model, alpha, "mondrian")
    cal_rate = (cp.gate_signals(cp.SplitConformal(alpha, True).fit(pc, run.y[s.cal])
                                .predict_sets(pc)) != 0).mean()
    tau = cp.matched_tau(pc, cal_rate)
    strategies = {
        "Always trade argmax": cp.argmax_signals(pt),
        f"Confidence gate (tau={tau:.2f}, matched rate)": cp.threshold_signals(pt, tau),
        f"Conformal gate (alpha={alpha:.2f})": conf_sig,
        "Adaptive conformal gate": gated_signals(res, model, alpha, "aci"),
    }
    rows = {}
    for name, sig in strategies.items():
        bt = run_backtest(res.data, s.test, sig, h, lat, fee)
        rows[name] = {"signal_rate": float((sig != 0).mean()), **bt.summary()}
    return pd.DataFrame(rows).T


def horizon_table(res: Results, model: str = "Gradient boosting") -> pd.DataFrame:
    """The accuracy illusion: headline accuracy vs whether argmax trading survives costs."""
    rows = []
    for h, run in res.runs.items():
        sig = cp.argmax_signals(run.probs_test[model])
        bt = run_backtest(res.data, res.split.test, sig, h, res.cfg.latency, res.cfg.fee_ticks)
        m = run.metrics.loc[model]
        rows.append({"horizon": h, "accuracy": m["accuracy"],
                     "prior_accuracy": run.metrics.loc["Class prior", "accuracy"],
                     "directional_precision": m["directional_precision"], **bt.summary()})
    return pd.DataFrame(rows).set_index("horizon")


def alpha_frontier(res: Results, model: str | None = None,
                   alphas=np.round(np.arange(0.05, 0.41, 0.05), 2),
                   latency: int | None = None, fee_ticks: float | None = None) -> pd.DataFrame:
    """Sweep the risk dial: how selectivity trades off against net edge per trade."""
    model = model or res.best_model
    lat = res.cfg.latency if latency is None else latency
    fee = res.cfg.fee_ticks if fee_ticks is None else fee_ticks
    run, s, h = res.primary, res.split, res.cfg.horizon
    pc, pt = run.probs_cal[model], run.probs_test[model]
    out = []
    for a in alphas:
        sig = gated_signals(res, model, float(a), "mondrian")
        rate_cal = (cp.gate_signals(cp.SplitConformal(float(a), True).fit(pc, run.y[s.cal])
                                    .predict_sets(pc)) != 0).mean()
        naive = cp.threshold_signals(pt, cp.matched_tau(pc, rate_cal))
        for name, sg in [("Conformal gate", sig), ("Confidence gate (matched)", naive)]:
            bt = run_backtest(res.data, s.test, sg, h, lat, fee).summary()
            out.append({"alpha": float(a), "strategy": name, "signal_rate": float((sg != 0).mean()), **bt})
    return pd.DataFrame(out)


def latency_curve(res: Results, model: str | None = None, alpha: float | None = None,
                  latencies=(0, 1, 2, 5, 10, 20, 40), fee_ticks: float | None = None) -> pd.DataFrame:
    model = model or res.best_model
    fee = res.cfg.fee_ticks if fee_ticks is None else fee_ticks
    alpha = res.cfg.alpha if alpha is None else alpha
    sig = gated_signals(res, model, alpha, "mondrian")
    rows = []
    for L in latencies:
        bt = run_backtest(res.data, res.split.test, sig, res.cfg.horizon, L, fee)
        rows.append({"latency_events": L, **bt.summary()})
    return pd.DataFrame(rows)


def regime_recovery(res: Results, model: str | None = None) -> dict | None:
    """Simulator only: does P(up) - P(down) track the *hidden* informed regime?"""
    if res.data.regime is None:
        return None
    model = model or res.best_model
    pt = res.primary.probs_test[model]
    score = pt[:, 2] - pt[:, 0]
    reg = res.data.regime[res.split.test].astype(float)
    by_regime = {int(r): float(score[reg == r].mean()) for r in (-1, 0, 1) if (reg == r).any()}
    return {"corr": float(np.corrcoef(score, reg)[0, 1]), "mean_score_by_regime": by_regime}
