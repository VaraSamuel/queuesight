"""Common limit order book container + loader for real LOBSTER data.

Everything downstream (features, labels, backtest) consumes `LOBData`, so the
synthetic simulator and real exchange data are interchangeable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LOBData:
    time: np.ndarray        # (T,) seconds
    ask_px: np.ndarray      # (T, L) level prices, best first
    ask_sz: np.ndarray      # (T, L)
    bid_px: np.ndarray      # (T, L)
    bid_sz: np.ndarray      # (T, L)
    event_type: np.ndarray  # (T,) 0 limit, 1 cancel, 2 market/execution
    event_side: np.ndarray  # (T,) +1 buy side, -1 sell side
    event_size: np.ndarray  # (T,)
    tick: float
    regime: np.ndarray | None = None  # latent informed regime (simulator only)

    def __len__(self) -> int:
        return len(self.time)

    @property
    def mid(self) -> np.ndarray:
        return 0.5 * (self.ask_px[:, 0] + self.bid_px[:, 0])

    @property
    def mid_ticks(self) -> np.ndarray:
        return self.mid / self.tick

    @property
    def n_levels(self) -> int:
        return self.ask_px.shape[1]

    def signed_trades(self) -> np.ndarray:
        """Aggressor-signed traded volume per event (+ = buyer initiated)."""
        return np.where(self.event_type == 2, self.event_side * self.event_size, 0.0)

    def slice(self, start: int, stop: int) -> "LOBData":
        s = slice(start, stop)
        return LOBData(
            self.time[s], self.ask_px[s], self.ask_sz[s], self.bid_px[s], self.bid_sz[s],
            self.event_type[s], self.event_side[s], self.event_size[s], self.tick,
            None if self.regime is None else self.regime[s],
        )


def load_lobster(message_csv: str, orderbook_csv: str, tick: float = 0.01) -> LOBData:
    """Load a LOBSTER (lobsterdata.com) message/orderbook file pair.

    LOBSTER prices are dollars * 10_000. Message event types:
    1 submit, 2 partial cancel, 3 delete, 4 visible execution, 5 hidden
    execution, 7 halt. For executions, direction -1 means a resting *sell* order
    was hit, i.e. the aggressor was a buyer.
    """
    msg = pd.read_csv(message_csv, header=None,
                      names=["time", "type", "order_id", "size", "price", "direction"])
    ob = pd.read_csv(orderbook_csv, header=None).to_numpy(dtype=np.float64)
    L = ob.shape[1] // 4
    keep = msg["type"].isin([1, 2, 3, 4]).to_numpy()
    msg, ob = msg[keep], ob[keep]

    etype = msg["type"].map({1: 0, 2: 1, 3: 1, 4: 2}).to_numpy(np.int8)
    direction = msg["direction"].to_numpy(np.int8)
    # for executions flip to aggressor side; for passive events keep order side
    side = np.where(etype == 2, -direction, direction).astype(np.int8)

    ask_px, ask_sz = ob[:, 0::4] / 10_000, ob[:, 1::4]
    bid_px, bid_sz = ob[:, 2::4] / 10_000, ob[:, 3::4]
    # LOBSTER pads empty levels with dummy prices (+-9999999999); zero their size
    ask_sz = np.where(ask_px > 1e5, 0.0, ask_sz)
    bid_sz = np.where(bid_px < 0, 0.0, bid_sz)
    assert ask_px.shape[1] == L
    return LOBData(
        time=msg["time"].to_numpy(np.float64),
        ask_px=ask_px, ask_sz=ask_sz, bid_px=bid_px, bid_sz=bid_sz,
        event_type=etype, event_side=side,
        event_size=msg["size"].to_numpy(np.float64), tick=tick,
    )
