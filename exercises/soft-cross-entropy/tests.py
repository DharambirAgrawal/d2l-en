import numpy as np

from submission import cross_entropy
from grader import expect_close, expect_true


def ref(logits, Y):
    P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return -(Y * np.log(P)).sum(1).mean()


def test_one_hot():
    """One-hot labels give the usual cross-entropy"""
    logits = np.array([[2.0, 1.0, 0.1], [0.5, 0.5, 3.0]])
    Y = np.eye(3)[[0, 2]]
    expect_close(cross_entropy(logits, Y), ref(logits, Y))


def test_soft():
    """Soft labels"""
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 2, (5, 4))
    Y = rng.random((5, 4)); Y /= Y.sum(1, keepdims=True)
    expect_close(cross_entropy(logits, Y), ref(logits, Y))


def test_extreme():
    """Huge logits: finite, no nan"""
    logits = np.array([[1000.0, 0.0, -1000.0]])
    Y = np.array([[1.0, 0.0, 0.0]])
    v = cross_entropy(logits, Y)
    expect_true(np.isfinite(v), f"loss is finite (got {v})")
    expect_close(v, 0.0, atol=1e-9)
    Y2 = np.array([[0.0, 0.0, 1.0]])
    expect_close(cross_entropy(logits, Y2), 2000.0)


def test_minimum():
    """The loss is smallest when the prediction equals the label distribution"""
    Y = np.array([[0.7, 0.2, 0.1]])
    best = cross_entropy(np.log(Y), Y)
    expect_close(best, -(Y * np.log(Y)).sum())
    expect_true(cross_entropy(np.log(Y) + np.array([[0.3, 0.0, -0.3]]), Y) > best,
                "any other prediction has a larger loss")
