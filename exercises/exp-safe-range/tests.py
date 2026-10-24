import numpy as np

from submission import exp_range, fits
from grader import expect_close, expect_equal


def test_ranges():
    """float16, float32, float64 ranges"""
    expect_close(exp_range(np.float16), (-9.704, 11.090), atol=1e-3)
    expect_close(exp_range(np.float32), (-87.337, 88.723), atol=1e-3)
    expect_close(exp_range(np.float64), (-708.396, 709.783), atol=1e-3)


def test_fits():
    """exp(100) overflows float32 but not float64; exp(-9) fits float16"""
    expect_equal(fits(100.0, np.float32), False)
    expect_equal(fits(100.0, np.float64), True)
    expect_equal(fits(-9.0, np.float16), True)
    expect_equal(fits(12.0, np.float16), False)
    expect_equal(fits(-100.0, np.float32), False)
