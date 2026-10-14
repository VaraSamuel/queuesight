"""Conformal prediction sets for 3-class mid-price forecasts, and trade gating.

The idea: a classifier's argmax says *which* way, but not *whether it knows*.
Conformal prediction turns any probabilistic forecaster into prediction sets
C(x) with a finite-sample guarantee P(y in C(x)) >= 1 - alpha (under
exchangeability). We only trade when the set is a single directional class --
{up} or {down} -- and abstain otherwise. Alpha becomes a risk dial with a
statistical meaning instead of an arbitrary confidence cutoff.

Order flow is not exchangeable (regimes drift), so we also implement Adaptive
Conformal Inference (Gibbs & Candes, 2021) with *delayed* feedback: the label
for a horizon-h forecast is only revealed h events later, so the online alpha
update may only use errors that have actually resolved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import DOWN, FLAT, UP


def nonconformity(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC score: 1 - p(true class). Low score = model agreed with reality."""
    return 1.0 - probs[np.arange(len(y)), y]


def conformal_quantile(sorted_scores: np.ndarray, alpha: float) -> float:
    n = len(sorted_scores)
    if alpha <= 0:
        return np.inf
    if alpha >= 1:
        return -np.inf
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(sorted_scores[k - 1])


@dataclass
class SplitConformal:
    alpha: float = 0.1
    mondrian: bool = False  # class-conditional coverage (one quantile per class)

    def fit(self, probs_cal: np.ndarray, y_cal: np.ndarray) -> SplitConformal:
        s = nonconformity(probs_cal, y_cal)
        if self.mondrian:
            self.qhat = np.array([conformal_quantile(np.sort(s[y_cal == c]), self.alpha)
                                  for c in range(probs_cal.shape[1])])
        else:
            self.qhat = np.full(probs_cal.shape[1], conformal_quantile(np.sort(s), self.alpha))
        return self

    def predict_sets(self, probs: np.ndarray) -> np.ndarray:
        return (1.0 - probs) <= self.qhat[None, :]


def adaptive_conformal(probs: np.ndarray, y: np.ndarray, cal_scores: np.ndarray,
                       alpha: float, gamma: float = 0.005, delay: int = 0):
    """Online ACI with label delay. Returns (sets, alpha_path)."""
    pool = np.sort(cal_scores)
    n = len(y)
    sets = np.zeros_like(probs, dtype=bool)
    miss = np.zeros(n, dtype=bool)
    alpha_path = np.empty(n)
    a = alpha
    for t in range(n):
        alpha_path[t] = a
        sets[t] = (1.0 - probs[t]) <= conformal_quantile(pool, a)
        miss[t] = not sets[t, y[t]]
        r = t - delay  # the forecast whose outcome becomes known now
        if r >= 0:
            a += gamma * (alpha - miss[r])
    return sets, alpha_path


def set_metrics(sets: np.ndarray, y: np.ndarray) -> dict:
    covered = sets[np.arange(len(y)), y]
    size = sets.sum(1)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(size.mean()),
        "singleton_rate": float((size == 1).mean()),
        "empty_rate": float((size == 0).mean()),
        **{f"coverage_{name}": float(covered[y == c].mean()) if (y == c).any() else np.nan
           for c, name in [(DOWN, "down"), (FLAT, "flat"), (UP, "up")]},
    }


def gate_signals(sets: np.ndarray) -> np.ndarray:
    """+1 iff the set is exactly {up}, -1 iff exactly {down}, else 0 (abstain)."""
    only = sets.sum(1) == 1
    return np.where(only & sets[:, UP], 1, np.where(only & sets[:, DOWN], -1, 0)).astype(np.int8)


def argmax_signals(probs: np.ndarray) -> np.ndarray:
    k = probs.argmax(1)
    return np.where(k == UP, 1, np.where(k == DOWN, -1, 0)).astype(np.int8)


def threshold_signals(probs: np.ndarray, tau: float) -> np.ndarray:
    """Naive baseline: trade argmax direction only when max prob >= tau."""
    sig = argmax_signals(probs)
    return np.where(probs.max(1) >= tau, sig, 0).astype(np.int8)


def matched_tau(probs_cal: np.ndarray, target_trade_rate: float) -> float:
    """Pick tau so the naive gate trades as often as a conformal gate (fair comparison)."""
    k = probs_cal.argmax(1)
    conf = np.where(k == FLAT, -np.inf, probs_cal.max(1))
    if target_trade_rate <= 0:
        return np.inf
    return float(np.quantile(conf, 1 - min(target_trade_rate, 1.0)))
