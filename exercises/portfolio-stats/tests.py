import numpy as np

from submission import portfolio_stats
from grader import expect_close, expect_raises, expect_true

MU = np.array([0.05, 0.10, 0.02])
SIGMA = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]])


def test_example():
    """Mean and variance of a mixed portfolio"""
    a = np.array([0.5, 0.3, 0.2])
    ret, var = portfolio_stats(a, MU, SIGMA)
    expect_close(ret, 0.5 * 0.05 + 0.3 * 0.10 + 0.2 * 0.02)
    expect_close(var, float(a @ SIGMA @ a))


def test_single_stock():
    """All-in on one stock gives that stock's mean and variance"""
    ret, var = portfolio_stats(np.array([0.0, 1.0, 0.0]), MU, SIGMA)
    expect_close(ret, 0.10)
    expect_close(var, 0.09)


def test_diversification_lowers_risk():
    """Splitting between two uncorrelated stocks lowers variance"""
    mu = np.array([0.1, 0.1])
    S = np.diag([0.04, 0.04])
    _, v_single = portfolio_stats(np.array([1.0, 0.0]), mu, S)
    _, v_split = portfolio_stats(np.array([0.5, 0.5]), mu, S)
    expect_true(v_split < v_single, 'An even split should have lower variance here.')


def test_weights_must_sum_to_one():
    """Weights not summing to 1 raise ValueError"""
    expect_raises(ValueError, portfolio_stats, np.array([0.5, 0.6, 0.0]), MU, SIGMA)
