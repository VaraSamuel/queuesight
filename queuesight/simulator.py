"""Queue-reactive limit order book simulator with a latent informed-flow regime.

The book lives on an integer tick grid. Each event is one of: limit order,
cancellation, or market order. Intensities react to the current queue state
(market orders lean toward the thin side of the book), and a hidden Markov
regime s in {-1, 0, +1} tilts order flow the way an informed trader would:
in a bullish regime there are more aggressive buys, more passive bids, and
more ask-side cancellations.

Because the regime is never observed, any predictive edge a model finds has to
come from *inferring* it out of order flow -- which is exactly the problem a
real LOB model faces. Ground-truth regimes are returned for diagnostics only.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .book import LOBData

GRID = 40_000


@dataclass
class SimConfig:
    n_events: int = 200_000
    n_levels: int = 10
    tick: float = 0.01
    start_price: float = 100.0
    seed: int = 7
    # event mix (probabilities per event, before reactive adjustments)
    p_market: float = 0.12
    p_limit: float = 0.50
    p_cancel: float = 0.38            # at target depth; scales with resting depth
    target_depth: float = 200.0       # total lots in the top n_levels, both sides
    # informed flow
    regime_switch: float = 1 / 600   # per-event probability of a regime change
    informed_strength: float = 0.18  # tilt of aggressive order direction
    book_tilt: float = 0.12          # tilt of passive placement / cancels
    imbalance_reaction: float = 0.15  # how much market orders chase L1 imbalance
    # sizes (geometric means, in lots)
    mean_limit_size: float = 3.0
    mean_cancel_size: float = 3.0
    mean_market_size: float = 4.0
    placement_decay: float = 0.45    # geometric decay of limit distance from touch
    p_improve: float = 0.60          # prob. a limit order improves a wide spread
    base_rate: float = 40.0          # events / second in a quiet regime

    def to_dict(self) -> dict:
        return asdict(self)


def _geom(rng: np.random.Generator, mean: float, n: int) -> np.ndarray:
    return rng.geometric(1.0 / max(mean, 1.0), size=n).astype(np.int64)


def simulate(cfg: SimConfig | None = None) -> LOBData:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    N, L = cfg.n_events, cfg.n_levels

    bid = np.zeros(GRID, dtype=np.int64)
    ask = np.zeros(GRID, dtype=np.int64)
    mid0 = GRID // 2
    bb, ba = mid0 - 1, mid0 + 1
    for k in range(40):
        bid[bb - k] = rng.integers(2, 12)
        ask[ba + k] = rng.integers(2, 12)

    # pre-drawn randomness (much faster than drawing inside the loop)
    u_type, u_side, u_aux, u_reg = (rng.random(N) for _ in range(4))
    lim_sz = _geom(rng, cfg.mean_limit_size, N)
    can_sz = _geom(rng, cfg.mean_cancel_size, N)
    mkt_sz = _geom(rng, cfg.mean_market_size, N)
    dist = np.minimum(rng.geometric(cfg.placement_decay, size=N) - 1, L - 1)
    new_regimes = rng.choice([-1, 0, 1], size=N, p=[0.3, 0.4, 0.3])
    dts = rng.exponential(1.0, size=N)

    out_bb = np.empty(N, dtype=np.int64)
    out_ba = np.empty(N, dtype=np.int64)
    out_bsz = np.empty((N, L), dtype=np.int32)
    out_asz = np.empty((N, L), dtype=np.int32)
    ev_type = np.empty(N, dtype=np.int8)    # 0 limit, 1 cancel, 2 market
    ev_side = np.empty(N, dtype=np.int8)    # +1 buy side, -1 sell side
    ev_size = np.empty(N, dtype=np.int32)
    regime_out = np.empty(N, dtype=np.int8)
    times = np.empty(N)

    regime, t = 0, 0.0
    for n in range(N):
        if u_reg[n] < cfg.regime_switch:
            regime = int(new_regimes[n])
        t += dts[n] / (cfg.base_rate * (1.0 + 0.6 * abs(regime)))

        qb, qa = bid[bb], ask[ba]
        imb = (qb - qa) / (qb + qa)
        # queue-reactive cancellation: the deeper the book, the more gets pulled
        depth = bid[bb - L + 1: bb + 1].sum() + ask[ba: ba + L].sum()
        w_can = cfg.p_cancel * depth / cfg.target_depth
        z = cfg.p_market + cfg.p_limit + w_can
        p_mkt, p_lim = cfg.p_market / z, cfg.p_limit / z
        r = u_type[n]

        if r < p_mkt:
            # aggressive order: informed tilt + chase the thin side
            p_buy = 0.5 + 0.5 * cfg.informed_strength * regime + cfg.imbalance_reaction * imb
            size = int(mkt_sz[n])
            ev_size[n] = size
            ev_type[n] = 2
            if u_side[n] < p_buy:
                ev_side[n] = 1
                while size > 0:
                    take = min(size, ask[ba])
                    ask[ba] -= take
                    size -= take
                    if ask[ba] == 0:
                        ba = _next_ask(ask, ba, rng)
            else:
                ev_side[n] = -1
                while size > 0:
                    take = min(size, bid[bb])
                    bid[bb] -= take
                    size -= take
                    if bid[bb] == 0:
                        bb = _next_bid(bid, bb, rng)

        elif r < p_mkt + p_lim:
            ev_type[n] = 0
            size = int(lim_sz[n])
            ev_size[n] = size
            spread = ba - bb
            improve = spread > 1 and u_aux[n] < cfg.p_improve
            if u_side[n] < 0.5 + 0.5 * cfg.book_tilt * regime:
                ev_side[n] = 1
                if improve:
                    bb += 1
                    bid[bb] += size
                else:
                    bid[bb - dist[n]] += size
            else:
                ev_side[n] = -1
                if improve:
                    ba -= 1
                    ask[ba] += size
                else:
                    ask[ba + dist[n]] += size

        else:
            ev_type[n] = 1
            # bullish informed flow pulls asks; bearish pulls bids
            if u_side[n] < 0.5 - 0.5 * cfg.book_tilt * regime:
                ev_side[n] = 1
                q = bid[bb - L + 1: bb + 1][::-1]
                lvl = _pick_weighted(q, u_aux[n])
                px = bb - lvl
                c = min(int(can_sz[n]), int(bid[px]))
                bid[px] -= c
                if px == bb and bid[bb] == 0:
                    bb = _next_bid(bid, bb, rng)
            else:
                ev_side[n] = -1
                q = ask[ba: ba + L]
                lvl = _pick_weighted(q, u_aux[n])
                px = ba + lvl
                c = min(int(can_sz[n]), int(ask[px]))
                ask[px] -= c
                if px == ba and ask[ba] == 0:
                    ba = _next_ask(ask, ba, rng)
            ev_size[n] = c

        if bb < 200 or ba > GRID - 200:
            raise RuntimeError("price walked off the simulation grid; lower n_events")

        out_bb[n], out_ba[n] = bb, ba
        out_bsz[n] = bid[bb - L + 1: bb + 1][::-1]
        out_asz[n] = ask[ba: ba + L]
        regime_out[n] = regime
        times[n] = t

    offs = np.arange(L)
    to_px = lambda ticks: cfg.start_price + (ticks - mid0) * cfg.tick  # noqa: E731
    return LOBData(
        time=times,
        ask_px=to_px(out_ba[:, None] + offs),
        ask_sz=out_asz.astype(np.float64),
        bid_px=to_px(out_bb[:, None] - offs),
        bid_sz=out_bsz.astype(np.float64),
        event_type=ev_type,
        event_side=ev_side,
        event_size=ev_size.astype(np.float64),
        tick=cfg.tick,
        regime=regime_out,
    )


def _pick_weighted(q: np.ndarray, u: float) -> int:
    total = q.sum()
    if total <= 0:
        return 0
    return int(min(np.searchsorted(np.cumsum(q), u * total, side="right"), len(q) - 1))


def _next_ask(ask: np.ndarray, ba: int, rng: np.random.Generator) -> int:
    for k in range(1, 60):
        if ask[ba + k] > 0:
            return ba + k
    # book exhausted: a liquidity provider steps in a few ticks away
    ba += 3
    ask[ba] = rng.integers(3, 10)
    return ba


def _next_bid(bid: np.ndarray, bb: int, rng: np.random.Generator) -> int:
    for k in range(1, 60):
        if bid[bb - k] > 0:
            return bb - k
    bb -= 3
    bid[bb] = rng.integers(3, 10)
    return bb
