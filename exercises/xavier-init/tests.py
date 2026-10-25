import numpy as np

from submission import layer_stds, xavier_normal
from grader import expect_close, expect_equal, expect_shape, expect_true


def test_shape_and_std():
    """Shape (n_in, n_out) and standard deviation sqrt(2 / (n_in + n_out))"""
    W = xavier_normal(300, 100, np.random.default_rng(0))
    expect_shape(W, (300, 100))
    expect_close(W.std(), np.sqrt(2 / 400), rtol=0.03)
    expect_close(W.mean(), 0.0, atol=0.003)


def test_uses_rng():
    """Uses the given generator (same seed, same weights)"""
    a = xavier_normal(5, 4, np.random.default_rng(7))
    b = xavier_normal(5, 4, np.random.default_rng(7))
    expect_close(a, b)


def test_layer_stds_linear():
    """With the identity activation, one layer's std is computed correctly"""
    rng = np.random.default_rng(1)
    X, W = rng.normal(size=(50, 8)), rng.normal(size=(8, 3))
    stds = layer_stds(X, [W], lambda z: z)
    expect_equal(len(stds), 1)
    expect_close(stds[0], (X @ W).std())


def test_vanish_and_survive():
    """Tiny weights make the signal vanish; Xavier keeps it alive through 30 tanh layers"""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(300, 128))
    small = layer_stds(X, [rng.normal(0, 0.01, (128, 128)) for _ in range(30)], np.tanh)
    xav = layer_stds(X, [xavier_normal(128, 128, rng) for _ in range(30)], np.tanh)
    expect_equal(len(xav), 30)
    expect_true(small[-1] < 1e-10, f'sigma = 0.01: last std {small[-1]:.2e} should vanish')
    expect_true(0.05 < xav[-1] < 0.9, f'Xavier: last std {xav[-1]:.3f} should stay in a healthy range')
