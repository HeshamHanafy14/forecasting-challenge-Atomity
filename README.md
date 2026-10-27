# Cloud Workload Forecasting

Forecast of aggregate CPU demand for the last 2 hours of an Azure VM trace
(Microsoft Azure Public Dataset V2, curated `data-v1` release), plus a simple
capacity-headroom rule built on the forecast. Task description:
[`ASSIGNMENT.md`](ASSIGNMENT.md), output format: [`SUBMISSION.md`](SUBMISSION.md).

## Result

| Metric (final 24 steps) | AR(1) | Persistence baseline |
|---|---|---|
| MAE (CPU %-points) | 0.129 | 0.147 |
| RMSE | 0.166 | 0.185 |

Relative MAE is 0.88 (12% lower than the baseline). Rolling-origin validation on
the training window gave 0.173 vs 0.189 (8% lower). The holdout window was
calmer than the validation windows, so the validation figure is the more
conservative estimate. Details, limitations and the capacity recommendation are
in the report ([`final_report.md`](reports/final_report.md),
[`final_report.pdf`](reports/final_report.pdf)); the step-by-step analysis is in
[`notebooks/eda.ipynb`](notebooks/eda.ipynb).

## Setup

Python 3.11 or newer.

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,analysis]"
```

On Linux/macOS, activate with `source .venv/bin/activate`.

## Data

Download `data-v1.tar.gz` from the
[release page](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1)
and extract it into `challenge_data/`:

```bash
tar -xzf data-v1.tar.gz -C challenge_data
cd challenge_data && sha256sum -c SHA256SUMS
```

The data files are not committed to this repository.

## Run

Generate `predictions.csv`:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

Add `--score` to also print the holdout error (diagnostic only).

Other commands:

```bash
python -m cwfc.capacity --data challenge_data    # headroom margins and coverage
python analysis/eda.py --data challenge_data     # EDA numbers and figures
python -m pytest -q                              # tests
```

The notebook `notebooks/eda.ipynb` walks through the same analysis, validation
and capacity rule. Open it with the `.venv` interpreter as the kernel (it runs
from the `notebooks/` folder).

## Method

- **Targets:** per timestamp, `active_vm_count` (distinct VMs with a valid
  reading) and `aggregate_cpu_utilization` (mean of VM `avg_cpu`). Missing
  readings are dropped, not filled; duplicate (VM, timestamp) pairs count once.
  The data had no gaps, duplicates or missing values.
- **Split:** the target series is built first, then the last 24 timestamps are
  held out and never touched for fitting, model choice or tuning.
- **`active_vm_count`:** constant at 10,000 in this release (every VM has a
  complete trace), so it is forecast by persistence.
- **CPU model:** candidates are persistence and ridge AR(p), p = 1, 2, 3. The
  choice uses rolling-origin validation (5 origins, 24-step horizon) inside the
  training window; AR(1) had the lowest MAE. SES and a moving average were tried
  during exploration and scored worse than persistence.
- **Capacity rule:** reserve forecast plus a margin, where the margin is a
  quantile of past under-forecast errors from the same validation rounds
  (`python -m cwfc.capacity`).

## Layout

```text
src/cwfc/       models, prediction pipeline, capacity rule, starter metrics
tests/          unit tests, output-contract and holdout-leakage tests
analysis/       EDA script (training window only)
notebooks/      analysis walkthrough notebook
reports/        final report (md and pdf) and figures
predictions.csv submission file (24 rows)
```

## How this was built

I developed this project with the help of Claude Code (Anthropic) as an AI coding
and tutoring assistant. I used it to learn the time-series methods involved, to
discuss modelling choices, and to draft code, tests and the report. I ran the code
myself, reviewed every result, and went through the analysis and decisions step by
step so that I can explain them.

## Limitations

- Only 134 timestamps (about 11 hours): no daily seasonality can be learned.
- VM churn is zero by construction of the release, so the count forecast says
  nothing about real-world churn.
- Few high-load steps fall inside the validation windows, so peak accuracy is
  not reliably measured.

Source data: [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset).
