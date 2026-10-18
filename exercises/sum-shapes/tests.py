import numpy as np

from submission import reduced_shape
from grader import expect_equal


def test_each_axis():
    """Shapes of sums along axes 0, 1 and 2 of a (2, 3, 4) tensor"""
    expect_equal(reduced_shape((2, 3, 4), 0), (3, 4))
    expect_equal(reduced_shape((2, 3, 4), 1), (2, 4))
    expect_equal(reduced_shape((2, 3, 4), 2), (2, 3))


def test_negative_axis():
    """axis=-1 means the last axis"""
    expect_equal(reduced_shape((2, 3, 4), -1), (2, 3))


def test_tuple_of_axes():
    """Reducing over several axes"""
    expect_equal(reduced_shape((2, 3, 4), (0, 2)), (3,))


def test_keepdims():
    """keepdims=True keeps reduced axes with length 1"""
    expect_equal(reduced_shape((2, 3, 4), 1, keepdims=True), (2, 1, 4))
    expect_equal(reduced_shape((2, 3, 4), None, keepdims=True), (1, 1, 1))


def test_all_axes():
    """axis=None reduces to a scalar"""
    expect_equal(reduced_shape((2, 3, 4), None), ())


def test_matches_numpy():
    """Agrees with NumPy for many cases"""
    X = np.zeros((2, 3, 4, 5))
    for axis in [0, 1, 2, 3, -1, -2, (0, 1), (1, 3), (0, 2, 3), None]:
        for keepdims in (False, True):
            expect_equal(reduced_shape(X.shape, axis, keepdims),
                         X.sum(axis=axis, keepdims=keepdims).shape,
                         f'Wrong shape for axis={axis}, keepdims={keepdims}.')
