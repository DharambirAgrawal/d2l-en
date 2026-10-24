import numpy as np

from submission import detect, importance_weights
from grader import expect_close, expect_true


def data(shift, seed=0, n=600):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, 4)), rng.normal(0, 1, (n, 4)) + shift


def test_no_shift():
    """Same distribution: accuracy near 0.5"""
    Xs, Xt = data(np.zeros(4), seed=1)
    acc, w, b = detect(Xs, Xt)
    expect_true(0.4 < acc < 0.6, f"accuracy near 0.5 (got {acc:.3f})")


def test_shift():
    """Shift in feature 2: detected, and the weight points at feature 2"""
    Xs, Xt = data(np.array([0.0, 0.0, 1.5, 0.0]), seed=2)
    acc, w, b = detect(Xs, Xt)
    expect_true(acc > 0.7, f"accuracy clearly above 0.5 (got {acc:.3f})")
    expect_true(np.argmax(np.abs(w)) == 2, f"largest weight on feature 2 (w = {np.round(w, 2)})")
    expect_true(w[2] > 0, "target has larger values of feature 2, so its weight is positive")


def test_weights():
    """Importance weights have mean 1 and reweight the source toward the target"""
    Xs, Xt = data(np.array([1.0, 0.0, 0.0, 0.0]), seed=3, n=2000)
    acc, w, b = detect(Xs, Xt)
    beta = importance_weights(Xs, w, b)
    expect_close(beta.mean(), 1.0)
    expect_true(np.all(beta > 0), "weights are positive")
    reweighted = np.average(Xs[:, 0], weights=beta)
    expect_true(abs(reweighted - Xt[:, 0].mean()) < 0.15,
                f"weighted source mean of feature 0 ({reweighted:.2f}) is close to the target's ({Xt[:, 0].mean():.2f})")
