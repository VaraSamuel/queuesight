"""Trade gating with a guarantee on the losing-trade rate (Learn then Test).

Conformal sets control label coverage, which is one step removed from money.
Here we control the quantity a desk cares about directly:

    P( trade loses money after costs | we traded )  <=  target

with probability >= 1 - delta over the calibration draw. This is the
Learn-then-Test framework (Angelopoulos, Bates, Candes, Jordan & Lei, 2021):
each candidate threshold lambda is a hypothesis test of
H0: loss_rate(lambda) > target. We test a small grid with a Bonferroni
correction and keep the valid threshold that trades most. (Fixed-sequence
testing avoids the correction but stops at its first failure, and the very top
of the conviction ranking is noisy enough that it often fails immediately.)

Two practical details:
  * Outcomes at nearby events overlap in time and are strongly dependent, so
    the calibration set is thinned to one decision per `horizon` events.
    Without that the binomial p-values are wildly overconfident.
  * The grid is defined by quantiles of the model's conviction
    |P(up) - P(down)|, which never looks at outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

from .features import DOWN, UP


def conviction(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    margin = probs[:, UP] - probs[:, DOWN]
    return np.abs(margin), np.sign(margin).astype(np.int8)


@dataclass
class LTTGate:
    target: float = 0.30      # max acceptable share of losing trades
    delta: float = 0.10       # failure probability of the guarantee
    trade_fractions: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)

    lam: float = np.inf
    trail: list | None = None

    def fit(self, probs_cal: np.ndarray, long_pnl: np.ndarray, short_pnl: np.ndarray,
            stride: int) -> LTTGate:
        strength, side = conviction(probs_cal[::stride])
        pnl = np.where(side > 0, long_pnl[::stride], short_pnl[::stride])
        lost = pnl < 0

        level = self.delta / len(self.trade_fractions)
        self.lam, self.trail = np.inf, []
        for frac in self.trade_fractions:
            lam = float(np.quantile(strength, 1 - frac))
            take = strength >= lam
            n, k = int(take.sum()), int(lost[take].sum())
            p_value = float(binom.cdf(k, n, self.target))  # small => loss rate below target
            valid = p_value <= level
            self.trail.append({"trade_fraction": frac, "lambda": lam, "trades": n,
                               "loss_rate": k / n, "p_value": p_value, "valid": valid})
            if valid:
                self.lam = min(self.lam, lam)              # most permissive valid threshold
        return self

    def signals(self, probs: np.ndarray) -> np.ndarray:
        strength, side = conviction(probs)
        return np.where(strength >= self.lam, side, 0).astype(np.int8)
