"""Multi-seed robustness study: are the headline effects real or one lucky path?

    python -m queuesight.robustness --seeds 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import ExperimentConfig, conformal_report, horizon_table, ltt_report, run_experiment, strategy_table
from .simulator import SimConfig

STRATS = {"Always trade argmax": "argmax", "Confidence gate": "confidence",
          "Conformal gate": "conformal", "Adaptive conformal gate": "aci",
          "Risk-controlled gate": "ltt"}


def one_seed(seed: int, events: int, alpha: float) -> dict:
    cfg = ExperimentConfig(sim=SimConfig(n_events=events, seed=seed), alpha=alpha, deep=False)
    res = run_experiment(cfg)
    model = "Gradient boosting"
    row = {"seed": seed}
    for h, r in horizon_table(res, model).iterrows():
        row[f"acc_h{h}"] = r["accuracy"]
        row[f"argmax_net_h{h}"] = r["net_pnl_per_trade"]
    st = strategy_table(res, model)
    for name, key in STRATS.items():
        match = [i for i in st.index if i.startswith(name)]
        row[f"{key}_net"] = st.loc[match[0], "net_pnl_per_trade"]
        row[f"{key}_trades"] = st.loc[match[0], "trades"]
    ltt = ltt_report(res, model)
    row["ltt_test_loss_rate"] = ltt["test_loss_rate"]
    row["ltt_guarantee_held"] = float(ltt["test_loss_rate"] <= ltt["target"]) if ltt["test_signal_rate"] > 0 else np.nan
    cov = conformal_report(res, model)
    row["split_cov"] = cov.iloc[0]["coverage"]
    row["mondrian_cov"] = cov.iloc[1]["coverage"]
    row["aci_cov"] = cov.iloc[2]["coverage"]
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--events", type=int, default=300_000)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    rows = []
    for s in range(1, a.seeds + 1):
        rows.append(one_seed(s, a.events, a.alpha))
        print(f"seed {s} done", flush=True)
    df = pd.DataFrame(rows).set_index("seed")
    diff = df["conformal_net"] - df["confidence_net"]
    df["conformal_minus_confidence"] = diff
    df["split_abs_cov_err"] = (df["split_cov"] - (1 - a.alpha)).abs()
    df["aci_abs_cov_err"] = (df["aci_cov"] - (1 - a.alpha)).abs()
    agg = pd.DataFrame({"mean": df.mean(), "std": df.std(), "min": df.min(), "max": df.max()})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(a.out) / "robustness_per_seed.csv")
    agg.to_csv(Path(a.out) / "robustness_summary.csv")
    pd.set_option("display.width", 200)
    print(agg.round(3))
    print(f"\nconformal > confidence in {(diff > 0).sum()}/{len(diff)} seeds")
    print(f"LTT > argmax in {(df['ltt_net'] > df['argmax_net']).sum()}/{len(df)} seeds, "
          f"> confidence gate in {(df['ltt_net'] > df['confidence_net']).sum()}/{len(df)}")


if __name__ == "__main__":
    main()
