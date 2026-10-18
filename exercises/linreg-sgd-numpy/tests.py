import numpy as np

from submission import train
from grader import expect_close, expect_true


def data(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = X @ np.array([2.0, -3.4]) + 4.2 + 0.01 * rng.standard_normal(n)
    return X, y


def test_recovers_parameters():
    """After 10 epochs, w is close to [2, -3.4] and b close to 4.2"""
    w, b = train(*data(), lr=0.03, epochs=10, batch_size=32)
    expect_close(w, np.array([2.0, -3.4]), atol=0.02)
    expect_close(b, 4.2, atol=0.02)


def test_three_epochs_like_the_book():
    """After 3 epochs (as in the section), errors are already below 0.3"""
    w, b = train(*data(), lr=0.03, epochs=3, batch_size=32)
    expect_close(w, np.array([2.0, -3.4]), atol=0.3)


def test_one_full_batch_step():
    """One full-batch step matches the gradient formula"""
    X, y = data(8)
    w, b = train(X, y, lr=0.1, epochs=1, batch_size=8)
    expect_close(w, 0.1 * X.T @ y / 8, rtol=1e-9)
    expect_close(b, 0.1 * y.mean(), rtol=1e-9)


def test_partial_last_batch():
    """Works when the batch size doesn't divide the data"""
    w, b = train(*data(103), lr=0.05, epochs=20, batch_size=10)
    expect_close(w, np.array([2.0, -3.4]), atol=0.05)


def test_too_large_learning_rate_diverges():
    """lr = 2.5 blows up (or goes non-finite)"""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        w, b = train(*data(), lr=2.5, epochs=3, batch_size=32)
    expect_true(not np.all(np.isfinite(w)) or np.abs(w).max() > 100,
                'With lr=2.5 the updates should overshoot and diverge.')
