"""Headroom rule: forecast plus a safety margin taken from validation errors.

    python -m cwfc.capacity --data challenge_data

The margin uses training data only. The holdout is used afterwards to show
how each margin would have performed.
"""
import argparse

import numpy as np

from cwfc.models import CANDIDATES
from cwfc.predict import CPU, HORIZON, build_targets, rolling_origins, select_model

QUANTILES = [0.5, 0.8, 0.9, 0.95, 0.99]


def validation_errors(train, func, kwargs):
    """actual - forecast for every step of every rolling-origin round."""
    errors = []
    for o in rolling_origins(len(train)):
        pred = func(train.iloc[:o], HORIZON, **kwargs)
        actual = train.iloc[o:o + HORIZON].to_numpy()
        errors.append(actual - pred)
    if not errors:
        raise ValueError("not enough training data for rolling validation")
    return np.array(errors)


def headroom_margin(errors, quantile):
    """Margin covering `quantile` of past cases; never below zero."""
    return max(float(np.quantile(errors, quantile)), 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    args = parser.parse_args()

    targets = build_targets(args.data)
    train = targets.iloc[:-HORIZON][CPU]
    holdout = targets.iloc[-HORIZON:][CPU].to_numpy()

    name, _ = select_model(train)
    func, kwargs = CANDIDATES[name]
    errors = validation_errors(train, func, kwargs)
    forecast = func(train, HORIZON, **kwargs)

    print(f"model: {name}")
    print("quantile  margin  covered  extra capacity (%)")
    for q in QUANTILES:
        margin = headroom_margin(errors, q)
        covered = (holdout <= forecast + margin).mean()
        extra = (margin / forecast).mean() * 100
        print(f"{q:<9} {margin:<7.3f} {covered:<8.2f} {extra:.2f}")


if __name__ == "__main__":
    main()
