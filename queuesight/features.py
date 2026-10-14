"""Order-flow features and mid-price movement labels.

Every feature at event t uses information available at or before t only.
References:
  * OFI -- Cont, Kukanov & Stoikov (2014), "The price impact of order book events"
  * multi-level OFI -- Xu, Gould & Howison (2018)
  * microprice -- Stoikov (2018), "The micro-price"
  * smoothed labels -- Ntakaris et al. (2018), FI-2010 benchmark
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .book import LOBData

WINDOWS = (10, 50, 200)
DOWN, FLAT, UP = 0, 1, 2
CLASS_NAMES = ("down", "flat", "up")


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = c[w:] - c[:-w]
    return np.concatenate([np.full(w - 1, np.nan), out])


def _lag_diff(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[w:] = x[w:] - x[:-w]
    return out


def multilevel_ofi(d: LOBData) -> np.ndarray:
    """Per-event order flow imbalance at each level, shape (T, L)."""
    bp, bs, ap, as_ = d.bid_px, d.bid_sz, d.ask_px, d.ask_sz
    e_b = np.zeros_like(bs)
    e_a = np.zeros_like(as_)
    pb, qb, pa, qa = bp[:-1], bs[:-1], ap[:-1], as_[:-1]
    e_b[1:] = (bp[1:] >= pb) * bs[1:] - (bp[1:] <= pb) * qb
    e_a[1:] = (ap[1:] <= pa) * as_[1:] - (ap[1:] >= pa) * qa
    return e_b - e_a


def build_features(d: LOBData) -> pd.DataFrame:
    tick = d.tick
    bs, as_ = d.bid_sz, d.ask_sz
    b1, a1 = bs[:, 0], as_[:, 0]
    mid = d.mid_ticks
    f: dict[str, np.ndarray] = {}

    f["spread"] = (d.ask_px[:, 0] - d.bid_px[:, 0]) / tick
    micro = (d.ask_px[:, 0] * b1 + d.bid_px[:, 0] * a1) / np.maximum(b1 + a1, 1e-9)
    f["microprice_dev"] = micro / tick - mid

    for depth in (1, 3, 5, d.n_levels):
        b, a = bs[:, :depth].sum(1), as_[:, :depth].sum(1)
        f[f"imb_L{depth}"] = (b - a) / np.maximum(b + a, 1e-9)

    # normalise flows by typical touch depth so features are scale free
    touch_depth = pd.Series(0.5 * (b1 + a1)).rolling(500, min_periods=1).mean().to_numpy()
    touch_depth = np.maximum(touch_depth, 1e-9)

    ofi = multilevel_ofi(d)
    ofi_l1, ofi_deep = ofi[:, 0], ofi[:, : min(5, d.n_levels)].sum(1)
    trades = d.signed_trades()
    for w in WINDOWS:
        f[f"ofi_L1_{w}"] = _rolling_sum(ofi_l1, w) / touch_depth
        f[f"ofi_L5_{w}"] = _rolling_sum(ofi_deep, w) / (5 * touch_depth)
        f[f"tfi_{w}"] = _rolling_sum(trades, w) / touch_depth
        f[f"ret_{w}"] = _lag_diff(mid, w)

    cancels = np.where(d.event_type == 1, d.event_side * d.event_size, 0.0)
    f["cancel_imb_50"] = -_rolling_sum(cancels, 50) / touch_depth  # bid pulls are bearish

    dt50 = _lag_diff(d.time, 50)
    f["log_intensity_50"] = -np.log(np.maximum(dt50, 1e-6) / 50)
    f["dq_bid1_10"] = _lag_diff(b1, 10) / touch_depth
    f["dq_ask1_10"] = _lag_diff(a1, 10) / touch_depth
    f["depth_ratio"] = np.log((bs.sum(1) + 1) / (as_.sum(1) + 1))
    f["vol_200"] = pd.Series(np.diff(mid, prepend=mid[0])).rolling(200).std().to_numpy()

    return pd.DataFrame(f)


def make_labels(d: LOBData, horizon: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """FI-2010 style smoothed label over the next `horizon` events.

    l_t = mean(mid_{t+1..t+h}) - mid_t, in ticks. Classes: down / flat / up using
    +-threshold. Returns (labels, l_t); the tail without a full horizon is -1.
    """
    mid = d.mid_ticks
    c = np.cumsum(np.concatenate([[0.0], mid]))
    T = len(mid)
    move = np.full(T, np.nan)
    fut_mean = (c[horizon + 1:] - c[1:T - horizon + 1]) / horizon
    move[: T - horizon] = fut_mean - mid[: T - horizon]
    y = np.full(T, -1, dtype=np.int64)
    ok = ~np.isnan(move)
    y[ok] = np.where(move[ok] > threshold, UP, np.where(move[ok] < -threshold, DOWN, FLAT))
    return y, move


def balanced_threshold(move: np.ndarray) -> float:
    """Threshold that puts roughly a third of samples in the flat class.

    At short horizons most mid moves are exactly zero, so the quantile can
    collapse to 0; a 0.1 tick floor keeps "flat" meaning "didn't really move".
    """
    return max(float(np.nanquantile(np.abs(move), 1 / 3)), 0.1)
