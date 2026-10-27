import numpy as np

from cwfc.capacity import headroom_margin


def test_margin_is_never_negative():
    errors = np.array([-0.5, -0.2, -0.1])
    assert headroom_margin(errors, 0.9) == 0.0


def test_margin_grows_with_quantile():
    errors = np.linspace(-1, 1, 101)
    low = headroom_margin(errors, 0.5)
    mid = headroom_margin(errors, 0.9)
    high = headroom_margin(errors, 0.99)
    assert low <= mid <= high
