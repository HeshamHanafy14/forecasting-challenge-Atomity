import numpy as np


def persistence(history, h):
    """Repeat the last known value h times."""
    return np.full(h, float(history.iloc[-1]))


def ar_forecast(history, h, p=1, ridge=1.0):
    """AR(p) model fitted on the history, then forecast h steps ahead."""
    y = history.to_numpy(dtype=float)
    if len(y) <= p:
        raise ValueError("not enough history for AR(p)")

    # each row of X is p consecutive values, target is the value that followed
    X = np.array([y[i:i + p] for i in range(len(y) - p)])
    target = y[p:]
    X = np.c_[np.ones(len(X)), X]

    # ridge regression: small penalty so the weights don't get extreme
    w = np.linalg.solve(X.T @ X + ridge * np.eye(p + 1), X.T @ target)

    # predict one step, then feed the prediction back in as the newest value
    window = list(y[-p:])
    forecast = []
    for _ in range(h):
        nxt = w[0] + np.dot(w[1:], window[-p:])
        forecast.append(nxt)
        window.append(nxt)
    return np.array(forecast)


# models we compare during validation
CANDIDATES = {
    "persistence": (persistence, {}),
    "ar1": (ar_forecast, {"p": 1}),
    "ar2": (ar_forecast, {"p": 2}),
    "ar3": (ar_forecast, {"p": 3}),
}
