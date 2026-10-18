import numpy as np

from submission import stack_and_recover
from grader import expect_close, expect_shape


def matrices():
    rng = np.random.default_rng(0)
    return [rng.normal(size=(100, 200)) for _ in range(3)]


def test_shape():
    """T has shape (100, 200, 3)"""
    T, _ = stack_and_recover(*matrices())
    expect_shape(T, (100, 200, 3))


def test_layers():
    """T[..., i] is the i-th matrix"""
    A, B, C = matrices()
    T, _ = stack_and_recover(A, B, C)
    expect_close(T[..., 0], A)
    expect_close(T[..., 1], B)
    expect_close(T[..., 2], C)


def test_recovers_b():
    """The slice recovers B"""
    A, B, C = matrices()
    _, B_again = stack_and_recover(A, B, C)
    expect_close(B_again, B)
