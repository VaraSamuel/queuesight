"""Execution-aware evaluation of directional signals.

A forecast that is "right" about the mid can still lose money: you buy at the
ask and sell at the bid. Each trade here:
  * is decided at event t,
  * is filled `latency` events later by crossing the spread (long @ ask, short @ bid),
  * is closed `horizon` events after entry by crossing again,
  * pays `fee_ticks` per side.
Trades never overlap (one unit position at a time), so PnL is actually attainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .book import LOBData


@dataclass
class BacktestResult:
    entries: np.ndarray       # event index of each fill
    sides: np.ndarray         # +1 long / -1 short
    pnl_net: np.ndarray       # ticks, after spread + fees
    pnl_mid: np.ndarray       # ticks, frictionless mid-to-mid (what accuracy "promises")

    @property
    def n(self) -> int:
        return len(self.pnl_net)

    def summary(self) -> dict:
        if self.n == 0:
            return {"trades": 0, "hit_rate": np.nan, "mid_pnl_per_trade": np.nan,
                    "net_pnl_per_trade": np.nan, "net_pnl_total": 0.0, "t_stat": np.nan}
        sd = self.pnl_net.std(ddof=1) if self.n > 1 else np.nan
        return {
            "trades": int(self.n),
            "hit_rate": float((self.pnl_net > 0).mean()),
            "mid_pnl_per_trade": float(self.pnl_mid.mean()),
            "net_pnl_per_trade": float(self.pnl_net.mean()),
            "net_pnl_total": float(self.pnl_net.sum()),
            "t_stat": float(self.pnl_net.mean() / sd * np.sqrt(self.n)) if sd and sd > 0 else np.nan,
        }


def run_backtest(d: LOBData, decision_idx: np.ndarray, signals: np.ndarray,
                 horizon: int, latency: int = 0, fee_ticks: float = 0.0) -> BacktestResult:
    ask, bid, mid = d.ask_px[:, 0] / d.tick, d.bid_px[:, 0] / d.tick, d.mid_ticks
    T = len(d)
    entries, sides, net, frictionless = [], [], [], []
    busy_until = -1
    for t, s in zip(decision_idx, signals, strict=True):
        if s == 0 or t <= busy_until:
            continue
        e, x = t + latency, t + latency + horizon
        if x >= T:
            break
        if s > 0:
            pnl = bid[x] - ask[e]
        else:
            pnl = bid[e] - ask[x]
        entries.append(e)
        sides.append(s)
        net.append(pnl - 2 * fee_ticks)
        frictionless.append(s * (mid[x] - mid[t]))
        busy_until = x
    return BacktestResult(np.array(entries, dtype=np.int64), np.array(sides, dtype=np.int8),
                          np.array(net, dtype=np.float64), np.array(frictionless, dtype=np.float64))


def trade_outcomes(d: LOBData, idx: np.ndarray, horizon: int, latency: int = 0,
                   fee_ticks: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Net PnL (ticks) of a hypothetical long and short decided at each index.

    Unlike `run_backtest` this ignores position overlap; it's what you need to
    calibrate a gate on per-decision outcomes.
    """
    ask, bid = d.ask_px[:, 0] / d.tick, d.bid_px[:, 0] / d.tick
    e = idx + latency
    x = e + horizon
    if x.max() >= len(d):
        raise ValueError("horizon + latency runs past the end of the data")
    long = bid[x] - ask[e] - 2 * fee_ticks
    short = bid[e] - ask[x] - 2 * fee_ticks
    return long, short
