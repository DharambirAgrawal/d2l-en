import numpy as np

from submission import softmax
from grader import expect_close, expect_shape, expect_true


def test_ordinary():
    """Matches the textbook formula on ordinary scores"""
    X = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    E = np.exp(X)
    expect_close(softmax(X), E / E.sum(1, keepdims=True))


def test_large():
    """Large scores (1000) don't overflow"""
    P = softmax(np.array([[1000.0, 1000.0, 0.0]]))
    expect_close(P, np.array([[0.5, 0.5, 0.0]]), atol=1e-12)


def test_very_negative():
    """Very negative scores (-1000) don't underflow to 0/0"""
    P = softmax(np.array([[-1000.0, -1001.0]]))
    expect_close(P, np.array([[1 / (1 + np.exp(-1)), np.exp(-1) / (1 + np.exp(-1))]]))


def test_rows():
    """Works row by row on a batch, rows sum to 1"""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 50, (6, 10))
    P = softmax(X)
    expect_shape(P, (6, 10))
    expect_true(np.all(np.isfinite(P)), "all outputs are finite")
    expect_close(P.sum(1), np.ones(6))
    expect_close(softmax(X + 7.0), P, message="adding a constant to a row must not change it")
