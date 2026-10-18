import numpy as np

from submission import numerical_gradient
from grader import expect_close, expect_shape


def test_exercise_7():
    """Gradient of 3 x1^2 + 5 e^x2 is [6 x1, 5 e^x2]"""
    f = lambda x: 3 * x[0] ** 2 + 5 * np.exp(x[1])
    for x in ([1.0, 0.0], [-2.0, 1.5], [0.3, -1.0]):
        x = np.array(x)
        expect_close(numerical_gradient(f, x), np.array([6 * x[0], 5 * np.exp(x[1])]), rtol=1e-5, atol=1e-6)


def test_exercise_8_norm():
    """Gradient of the l2 norm is x / ||x||"""
    x = np.array([3.0, -4.0, 12.0])
    expect_close(numerical_gradient(np.linalg.norm, x), x / 13.0, rtol=1e-5, atol=1e-7)


def test_shape_and_input_unchanged():
    """Same shape as x, and x is not modified"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    before = x.copy()
    g = numerical_gradient(lambda v: (v ** 2).sum(), x)
    expect_shape(g, (4,))
    expect_close(g, 2 * before, rtol=1e-6, atol=1e-6)
    expect_close(x, before, message='numerical_gradient modified x.')


def test_quadratic_form():
    """Gradient of x^T A x is (A + A^T) x"""
    rng = np.random.default_rng(0)
    A = rng.normal(size=(3, 3))
    x = rng.normal(size=3)
    expect_close(numerical_gradient(lambda v: v @ A @ v, x), (A + A.T) @ x, rtol=1e-5, atol=1e-6)
