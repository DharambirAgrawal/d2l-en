import numpy as np

from submission import huber, huber_grad
from grader import expect_close, expect_true


def test_values():
    """Quadratic inside, linear outside"""
    expect_close(huber(np.array([0.5]), np.array([0.0])), 0.125)
    expect_close(huber(np.array([3.0]), np.array([0.0])), 2.5)
    expect_close(huber(np.array([0.5, 3.0]), np.array([0.0, 0.0])), 1.3125)


def test_continuous_at_threshold():
    """Both pieces meet at |error| = sigma"""
    for s in (0.5, 1.0, 2.0):
        a = huber(np.array([s - 1e-9]), np.array([0.0]), s)
        b = huber(np.array([s + 1e-9]), np.array([0.0]), s)
        expect_close(a, b, atol=1e-6)


def test_gradient():
    """Gradient matches a numerical gradient"""
    rng = np.random.default_rng(0)
    y_hat, y = rng.normal(0, 3, 8), rng.normal(0, 1, 8)
    num = np.zeros(8)
    for i in range(8):
        e = np.zeros(8); e[i] = 1e-6
        num[i] = (huber(y_hat + e, y, 1.5) - huber(y_hat - e, y, 1.5)) / 2e-6
    expect_close(huber_grad(y_hat, y, 1.5), num, rtol=1e-4, atol=1e-7)


def test_bounded_gradient():
    """Huge errors have gradient magnitude at most 1/n"""
    g = huber_grad(np.array([1e6, -1e6]), np.array([0.0, 0.0]))
    expect_close(np.abs(g), np.array([0.5, 0.5]))
