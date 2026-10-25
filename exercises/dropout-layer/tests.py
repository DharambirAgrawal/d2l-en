import numpy as np

from submission import dropout
from grader import expect_close, expect_shape, expect_true


def test_eval_mode():
    """At test time the input passes through unchanged"""
    X = np.random.default_rng(0).normal(size=(4, 5))
    expect_close(dropout(X, 0.5, False, np.random.default_rng(1)), X)


def test_extremes():
    """p = 0 changes nothing; p = 1 returns zeros"""
    X = np.ones((3, 4))
    expect_close(dropout(X, 0.0, True, np.random.default_rng(0)), X)
    out = dropout(X, 1.0, True, np.random.default_rng(0))
    expect_shape(out, (3, 4))
    expect_close(out, np.zeros((3, 4)))


def test_values():
    """Survivors are scaled by 1/(1-p), the rest are exactly 0"""
    X = np.full((200, 50), 2.0)
    out = dropout(X, 0.3, True, np.random.default_rng(3))
    vals = set(np.round(np.unique(out), 6))
    expect_true(vals <= {0.0, round(2.0 / 0.7, 6)}, f'values should be 0 or {2 / 0.7:.4f}, got {sorted(vals)[:5]}')
    expect_close((out == 0).mean(), 0.3, atol=0.02, message='fraction dropped')


def test_expectation():
    """The mean activation is unchanged (unbiased)"""
    X = np.random.default_rng(4).uniform(1, 3, (400, 400))
    out = dropout(X, 0.6, True, np.random.default_rng(5))
    expect_close(out.mean(), X.mean(), rtol=0.02)
