import math

from submission import tangent_line
from grader import expect_close


def test_exercise_6():
    """x^3 - 1/x at x = 1 gives y = 4x - 4"""
    slope, intercept = tangent_line(lambda x: x ** 3 - 1 / x, 1.0)
    expect_close(slope, 4.0, rtol=1e-5)
    expect_close(intercept, -4.0, rtol=1e-5)


def test_section_example():
    """3x^2 - 4x at x = 1 gives y = 2x - 3 (from the section)"""
    slope, intercept = tangent_line(lambda x: 3 * x ** 2 - 4 * x, 1.0)
    expect_close(slope, 2.0, rtol=1e-5)
    expect_close(intercept, -3.0, rtol=1e-5)


def test_sine():
    """sin at pi/2 has a flat tangent y = 1"""
    slope, intercept = tangent_line(math.sin, math.pi / 2)
    expect_close(slope, 0.0, atol=1e-6)
    expect_close(intercept, 1.0, rtol=1e-6)
