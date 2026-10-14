"""Forecasters with a shared interface: fit(idx, y) / predict_proba(idx).

Models address samples by *event index* so tabular models (hand-crafted
order-flow features) and the sequence model (raw book snapshots) can be
trained, calibrated and backtested on exactly the same rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .book import LOBData


class TabularModel:
    def __init__(self, name: str, estimator, features: pd.DataFrame):
        self.name = name
        self.est = estimator
        self.X = features.to_numpy(np.float64)
        self.feature_names = list(features.columns)

    def fit(self, idx: np.ndarray, y: np.ndarray) -> TabularModel:
        self.est.fit(self.X[idx], y)
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        return self.est.predict_proba(self.X[idx])


def logistic(features: pd.DataFrame) -> TabularModel:
    est = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500))
    return TabularModel("Logistic (OFI)", est, features)


def boosted(features: pd.DataFrame, seed: int = 0) -> TabularModel:
    est = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=200,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=seed,
    )
    return TabularModel("Gradient boosting", est, features)


class DeepLOBLite:
    """Compact conv + GRU over raw snapshots, in the spirit of DeepLOB (Zhang et al. 2019).

    Input per sample: the last `window` events x (4 * L + 1) channels -- level
    prices relative to mid (ticks), log sizes normalised by touch depth, and the
    mid path relative to the decision time.
    """

    name = "DeepLOB-lite (raw book)"

    def __init__(self, d: LOBData, window: int = 50, epochs: int = 4,
                 max_train: int = 60_000, seed: int = 0):
        import torch  # optional dependency, imported lazily

        self.torch = torch
        self.window, self.epochs, self.max_train, self.seed = window, epochs, max_train, seed
        mid = d.mid_ticks[:, None]
        depth = pd.Series(0.5 * (d.bid_sz[:, 0] + d.ask_sz[:, 0])).rolling(500, min_periods=1).mean()
        depth = np.maximum(depth.to_numpy()[:, None], 1e-9)
        self.raw = np.concatenate([
            d.ask_px / d.tick - mid, np.log1p(d.ask_sz / depth),
            d.bid_px / d.tick - mid, np.log1p(d.bid_sz / depth),
        ], axis=1).astype(np.float32)
        self.mid = d.mid_ticks.astype(np.float32)
        self.net = None

    def _batch(self, idx: np.ndarray):
        offs = np.arange(-self.window + 1, 1)
        rows = idx[:, None] + offs
        x = self.raw[rows]
        path = (self.mid[rows] - self.mid[idx][:, None])[..., None]
        return self.torch.from_numpy(np.concatenate([x, path], axis=2))

    def _build(self, c_in: int):
        nn = self.torch.nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(c_in, 32, 5, padding=2), nn.GELU(),
                    nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
                )
                self.gru = nn.GRU(32, 48, batch_first=True)
                self.head = nn.Linear(48, 3)

            def forward(self, x):                       # x: (B, W, C)
                h = self.conv(x.transpose(1, 2)).transpose(1, 2)
                _, last = self.gru(h)
                return self.head(last[-1])

        return Net()

    def fit(self, idx: np.ndarray, y: np.ndarray) -> DeepLOBLite:
        torch = self.torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        keep = idx >= self.window
        idx, y = idx[keep], y[keep]
        if len(idx) > self.max_train:
            pick = np.sort(rng.choice(len(idx), self.max_train, replace=False))
            idx, y = idx[pick], y[pick]
        self.net = self._build(self.raw.shape[1] + 1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()
        yt = torch.from_numpy(y.astype(np.int64))
        self.net.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(idx))
            for s in range(0, len(order), 256):
                b = order[s:s + 256]
                opt.zero_grad()
                loss = loss_fn(self.net(self._batch(idx[b])), yt[b])
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, idx: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                out.append(torch.softmax(self.net(self._batch(idx[s:s + 2048])), 1).numpy())
        p = np.concatenate(out).astype(np.float64)
        return p / p.sum(1, keepdims=True)
