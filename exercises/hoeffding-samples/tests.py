import math

from submission import error_bar, samples_needed
from grader import expect_close, expect_equal, expect_true


def test_known():
    """Small cases"""
    expect_equal(samples_needed(0.01, 0.05), 18445)
    expect_equal(samples_needed(0.1, 0.05), 185)


def test_exercise():
    """Exercise 1: t = 0.0001 at 99.9%"""
    n = samples_needed(1e-4, 1e-3)
    expect_true(3.80e8 < n < 3.81e8, f"about 380 million (got {n})")


def test_smallest():
    """n is the smallest that works"""
    for t, d in ((0.05, 0.01), (0.02, 0.1)):
        n = samples_needed(t, d)
        expect_true(2 * math.exp(-2 * n * t * t) <= d, "n works")
        expect_true(2 * math.exp(-2 * (n - 1) * t * t) > d, "n - 1 does not")


def test_error_bar():
    """error_bar inverts samples_needed"""
    expect_close(error_bar(10000, 0.001), 0.019495, atol=1e-6)
    expect_close(error_bar(4 * 10000, 0.001), 0.019495 / 2, atol=1e-6)
