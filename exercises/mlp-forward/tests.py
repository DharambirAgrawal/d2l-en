import numpy as np

from submission import count_params, mlp_forward
from grader import expect_close, expect_equal, expect_shape


def make(sizes, seed=0):
    rng = np.random.default_rng(seed)
    return [(rng.normal(0, 0.5, (a, b)), rng.normal(0, 0.5, b)) for a, b in zip(sizes[:-1], sizes[1:])]


def test_one_layer_is_linear():
    """With one layer there is no ReLU: plain X @ W + b"""
    (W, b), = make([5, 3])
    X = np.random.default_rng(1).normal(size=(4, 5))
    expect_close(mlp_forward(X, [(W, b)]), X @ W + b)


def test_two_layers():
    """Two layers: ReLU on the hidden layer only (logits may be negative)"""
    p = make([5, 7, 3])
    X = np.random.default_rng(2).normal(size=(6, 5))
    want = np.maximum(X @ p[0][0] + p[0][1], 0) @ p[1][0] + p[1][1]
    out = mlp_forward(X, p)
    expect_shape(out, (6, 3))
    expect_close(out, want)


def test_deep():
    """Four layers"""
    p = make([3, 6, 6, 6, 2], seed=3)
    X = np.random.default_rng(4).normal(size=(10, 3))
    H = X
    for W, b in p[:-1]:
        H = np.maximum(H @ W + b, 0)
    expect_close(mlp_forward(X, p), H @ p[-1][0] + p[-1][1])


def test_count():
    """Parameter counts"""
    expect_equal(count_params([784, 256, 10]), 203530)
    expect_equal(count_params([784, 256, 256, 10]), 203530 + 65792)
    expect_equal(count_params([784, 10]), 7850)
