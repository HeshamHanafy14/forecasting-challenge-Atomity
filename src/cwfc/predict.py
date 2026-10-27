"""Generate predictions.csv for the last 24 timestamps.

    python -m cwfc.predict --data challenge_data --output predictions.csv

The holdout is cut off first. The model is chosen with rolling-origin
validation inside the training part only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cwfc.metrics import mae, rmse
from cwfc.models import CANDIDATES, persistence

HORIZON = 24        # 24 x 5 min = 2 hours
ORIGIN_STEP = 6     # distance between validation starting points
N_ORIGINS = 5
MIN_HISTORY = 12

CPU = "aggregate_cpu_utilization"
COUNT = "active_vm_count"


def build_targets(data_dir):
    """One row per timestamp: number of VMs reporting and their mean avg_cpu."""
    cpu = pd.read_csv(
        Path(data_dir) / "vm_cpu_readings.csv.gz",
        usecols=["timestamp", "vm_id", "avg_cpu"],
    )
    # missing readings are dropped, not filled in; duplicates counted once
    cpu = cpu.dropna(subset=["avg_cpu"]).drop_duplicates(["vm_id", "timestamp"])

    by_time = cpu.groupby("timestamp")
    targets = pd.DataFrame({
        COUNT: by_time["vm_id"].nunique(),
        CPU: by_time["avg_cpu"].mean(),
    })
    return targets.sort_index()


def rolling_origins(n_train):
    """Starting points for validation. Every window ends inside the training data."""
    last = n_train - HORIZON
    first = last - (N_ORIGINS - 1) * ORIGIN_STEP
    return [o for o in range(first, last + 1, ORIGIN_STEP) if o >= MIN_HISTORY]


def select_model(series):
    """Return the candidate with the lowest average MAE over the origins."""
    origins = rolling_origins(len(series))
    if not origins:
        return "persistence", {}

    scores = {}
    for name, (func, kwargs) in CANDIDATES.items():
        errors = []
        for o in origins:
            pred = func(series.iloc[:o], HORIZON, **kwargs)
            actual = series.iloc[o:o + HORIZON].to_numpy()
            errors.append(mae(actual, pred))
        scores[name] = float(np.mean(errors))
    return min(scores, key=scores.get), scores


def run(data_dir, output, score=False):
    targets = build_targets(data_dir)
    if len(targets) <= HORIZON:
        raise ValueError(f"need more than {HORIZON} timestamps, got {len(targets)}")

    train = targets.iloc[:-HORIZON]
    holdout = targets.iloc[-HORIZON:]

    name, scores = select_model(train[CPU])
    func, kwargs = CANDIDATES[name]
    cpu_pred = np.clip(func(train[CPU], HORIZON, **kwargs), 0, 100)
    # the VM count is constant in this data, so repeat the last value
    count_pred = np.maximum(persistence(train[COUNT], HORIZON), 0)

    predictions = pd.DataFrame({
        "timestamp": holdout.index,
        COUNT: count_pred,
        CPU: cpu_pred,
    })
    predictions.to_csv(output, index=False)

    print("rolling-origin MAE:", scores)
    print("selected model:", name)
    if score:
        # only for checking, run once after the model is fixed
        y = holdout[CPU].to_numpy()
        base_pred = persistence(train[CPU], HORIZON)
        print(f"holdout MAE={mae(y, cpu_pred):.4f} RMSE={rmse(y, cpu_pred):.4f}")
        print(f"baseline (persistence) MAE={mae(y, base_pred):.4f} RMSE={rmse(y, base_pred):.4f}")
        print(f"relative MAE={mae(y, cpu_pred) / mae(y, base_pred):.3f}")
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--score", action="store_true", help="print holdout error")
    args = parser.parse_args()
    run(args.data, args.output, args.score)


if __name__ == "__main__":
    main()
