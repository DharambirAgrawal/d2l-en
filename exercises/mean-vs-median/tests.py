import numpy as np

from submission import best_absolute, best_squared
from grader import expect_close, expect_true


def loss_sq(x, b):
    return ((x - b) ** 2).sum()


def loss_abs(x, b):
    return np.abs(x - b).sum()


def test_squared_is_optimal():
    """No other b has lower squared loss"""
    x = np.random.default_rng(0).normal(5, 2, 101)
    b = best_squared(x)
    for d in (-0.01, 0.01, -1, 1):
        expect_true(loss_sq(x, b) <= loss_sq(x, b + d) + 1e-9, f'b + {d} has lower squared loss.')


def test_absolute_is_optimal():
    """No other b has lower absolute loss"""
    x = np.random.default_rng(1).exponential(3, 101)
    b = best_absolute(x)
    for d in (-0.01, 0.01, -1, 1):
        expect_true(loss_abs(x, b) <= loss_abs(x, b + d) + 1e-9, f'b + {d} has lower absolute loss.')


def test_values():
    """Mean and median of a small example"""
    x = np.array([1.0, 2.0, 6.0])
    expect_close(best_squared(x), 3.0)
    expect_close(best_absolute(x), 2.0)


def test_outlier_robustness():
    """The absolute-loss estimate barely moves with an outlier"""
    x = np.array([2.0, 3.0, 3.5, 4.0])
    y = np.append(x, 1000.0)
    expect_true(abs(best_absolute(y) - best_absolute(x)) < 1, 'The median should stay close.')
    expect_true(abs(best_squared(y) - best_squared(x)) > 100, 'The mean should move a lot.')
