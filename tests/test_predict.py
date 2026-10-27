import numpy as np
import pandas as pd
import pytest

from cwfc.predict import (
    COUNT,
    CPU,
    HORIZON,
    build_targets,
    rolling_origins,
    run,
)

N_VMS = 5
N_STEPS = 60


def make_data(folder, holdout_shift=0.0):
    """Small synthetic release. holdout_shift moves only the last HORIZON steps."""
    rng = np.random.default_rng(0)
    rows = []
    for v in range(N_VMS):
        for t in range(N_STEPS):
            avg = rng.uniform(5, 15)
            if t >= N_STEPS - HORIZON:
                avg += holdout_shift
            rows.append({"timestamp": t * 300, "vm_id": f"vm{v}",
                         "min_cpu": 1.0, "max_cpu": 20.0, "avg_cpu": avg})
    pd.DataFrame(rows).to_csv(folder / "vm_cpu_readings.csv.gz", index=False)
    return folder


def test_build_targets_counts_each_vm_once_per_timestamp(tmp_path):
    targets = build_targets(make_data(tmp_path))
    assert len(targets) == N_STEPS
    assert (targets[COUNT] == N_VMS).all()
    assert targets[CPU].between(0, 100).all()


def test_predictions_follow_submission_contract(tmp_path):
    data = make_data(tmp_path)
    out = tmp_path / "predictions.csv"
    run(data, out)

    pred = pd.read_csv(out)
    expected_ts = build_targets(data).index[-HORIZON:]

    assert list(pred.columns) == ["timestamp", COUNT, CPU]
    assert len(pred) == HORIZON
    assert (pred["timestamp"].to_numpy() == expected_ts.to_numpy()).all()
    assert pred["timestamp"].is_unique
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred[COUNT] >= 0).all()
    assert pred[CPU].between(0, 100).all()


def test_predictions_do_not_depend_on_holdout_values(tmp_path):
    folder_a = tmp_path / "a"
    folder_b = tmp_path / "b"
    folder_a.mkdir()
    folder_b.mkdir()
    make_data(folder_a)
    make_data(folder_b, holdout_shift=30.0)

    pred_a = run(folder_a, folder_a / "p.csv")
    pred_b = run(folder_b, folder_b / "p.csv")
    pd.testing.assert_frame_equal(pred_a, pred_b)


def test_rolling_origins_stay_inside_training_window():
    n_train = 110
    for origin in rolling_origins(n_train):
        assert origin + HORIZON <= n_train


def test_too_little_data_raises(tmp_path):
    rows = [{"timestamp": t * 300, "vm_id": "vm0", "min_cpu": 1.0,
             "max_cpu": 2.0, "avg_cpu": 1.5} for t in range(HORIZON)]
    pd.DataFrame(rows).to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "p.csv")
