import numpy as np
import pandas as pd
import pytest

from cwfc.models import ar_forecast, persistence


def test_persistence_repeats_last_value():
    history = pd.Series([1.0, 2.0, 3.0])
    assert list(persistence(history, 4)) == [3.0] * 4


def test_ar_forecast_length_and_values_are_finite():
    history = pd.Series(9.0 + 0.5 * np.sin(np.arange(60) / 5))
    pred = ar_forecast(history, 24, p=2)
    assert len(pred) == 24
    assert np.isfinite(pred).all()


def test_ar_forecast_of_constant_series_stays_constant():
    history = pd.Series([5.0] * 30)
    pred = ar_forecast(history, 10, p=1, ridge=1e-9)
    assert np.allclose(pred, 5.0, atol=1e-3)


def test_ar_forecast_needs_enough_history():
    with pytest.raises(ValueError):
        ar_forecast(pd.Series([1.0]), 3, p=1)
