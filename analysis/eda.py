"""Exploratory analysis on the training window only.

    python analysis/eda.py --data challenge_data

Prints the numbers used in the report and saves figures to reports/figures.
The final 24 timestamps (holdout) are not used here.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cwfc.predict import COUNT, CPU, HORIZON, build_targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--out", default="reports/figures")
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = build_targets(data)
    train = targets.iloc[:-HORIZON]
    train_end = train.index[-1]
    series = train[CPU]

    cpu = pd.read_csv(data / "vm_cpu_readings.csv.gz",
                      usecols=["timestamp", "vm_id", "avg_cpu"])
    cpu = cpu[cpu["timestamp"] <= train_end]
    meta = pd.read_csv(data / "vm_metadata.csv.gz")

    print("== Target series (training window) ==")
    print("timestamps:", len(train), "| step (s):", sorted(set(train.index.to_series().diff().dropna())))
    print("active_vm_count min/max:", train[COUNT].min(), train[COUNT].max())
    print(f"mean CPU: {series.mean():.3f}  std: {series.std():.3f}")
    print(f"peak: {series.max():.3f} at {series.idxmax()}  low: {series.min():.3f} at {series.idxmin()}")
    print(f"90% line: {series.quantile(0.9):.3f}")

    vm_avg = cpu.groupby("vm_id")["avg_cpu"].mean().rename("vm_mean_cpu")
    print("\n== Per-VM mean CPU (training window) ==")
    print(vm_avg.describe())
    print(vm_avg.quantile([0.5, 0.9, 0.95, 0.99]))
    print("share of VMs under 5%:", round((vm_avg < 5).mean(), 4))

    joined = meta.set_index("vm_id").join(vm_avg)
    print("\n== By VM category ==")
    print(joined.groupby("vm_category")["vm_mean_cpu"].describe())
    print("\n== VM sizes ==")
    print(meta["virtual_core_bucket"].value_counts())

    print("\n== Lifetimes ==")
    print("created min/max:", meta["timestamp_vm_created"].min(), meta["timestamp_vm_created"].max())
    print("earliest deletion:", meta["timestamp_vm_deleted"].min(), "| training window ends:", train_end)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    series.plot(ax=ax)
    ax.axhline(series.quantile(0.9), color="grey", linestyle="--", label="90% line")
    ax.set_title("Mean CPU across all VMs (training window)")
    ax.set_xlabel("timestamp (s)")
    ax.set_ylabel("avg CPU (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "series.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(vm_avg, bins=60, log=True)
    ax.set_title("Mean CPU per VM (log count)")
    ax.set_xlabel("mean avg CPU (%)")
    ax.set_ylabel("VMs")
    fig.tight_layout()
    fig.savefig(out / "vm_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    joined.groupby("vm_category")["vm_mean_cpu"].agg(["median", "mean"]).plot.bar(ax=ax)
    ax.set_title("Mean CPU per VM by category")
    ax.set_ylabel("avg CPU (%)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out / "by_category.png", dpi=150)

    print("\nfigures saved to", out)


if __name__ == "__main__":
    main()
