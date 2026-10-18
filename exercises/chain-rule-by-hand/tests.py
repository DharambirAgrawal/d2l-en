import math

from submission import f_and_grad
from grader import expect_close


def reference(x):
    return (math.log(x ** 2) * math.sin(x) + 1 / x,
            2 * math.sin(x) / x + math.log(x ** 2) * math.cos(x) - 1 / x ** 2)


def test_value():
    """f(x) is correct"""
    for x in (0.5, 1.0, 2.0, 3.7):
        expect_close(f_and_grad(x)[0], reference(x)[0], rtol=1e-9)


def test_derivative():
    """f'(x) matches the analytic derivative"""
    for x in (0.5, 1.0, 2.0, 3.7):
        expect_close(f_and_grad(x)[1], reference(x)[1], rtol=1e-9)


def test_matches_numerical():
    """f'(x) agrees with a central difference"""
    h = 1e-6
    for x in (0.8, 2.5):
        num = (f_and_grad(x + h)[0] - f_and_grad(x - h)[0]) / (2 * h)
        expect_close(f_and_grad(x)[1], num, rtol=1e-6)
