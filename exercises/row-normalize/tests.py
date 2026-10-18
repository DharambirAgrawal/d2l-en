import numpy as np

from submission import row_normalize
from grader import expect_close, expect_shape


def test_example():
    """A non-square (3, 2) matrix"""
    A = np.array([[1., 3.], [2., 2.], [0., 5.]])
    expect_close(row_normalize(A), np.array([[0.25, 0.75], [0.5, 0.5], [0., 1.]]))


def test_square_matrix():
    """A square matrix (where A / A.sum(axis=1) silently gives a wrong answer)"""
    A = np.array([[1., 1.], [1., 3.]])
    expect_close(row_normalize(A), np.array([[0.5, 0.5], [0.25, 0.75]]))


def test_rows_sum_to_one():
    """Random matrices: every row sums to 1, shape unchanged"""
    rng = np.random.default_rng(1)
    A = rng.uniform(0.1, 5, size=(7, 4))
    out = row_normalize(A)
    expect_shape(out, (7, 4))
    expect_close(out.sum(axis=1), np.ones(7))


def test_integer_input():
    """Integer counts become fractions"""
    expect_close(row_normalize(np.array([[1, 1, 2]])), np.array([[0.25, 0.25, 0.5]]))
