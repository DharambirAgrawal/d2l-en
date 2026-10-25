from submission import early_stop
from grader import expect_equal


def test_example():
    """Stops 3 epochs after the best (0.69 at epoch 4), before the late 0.6"""
    expect_equal(early_stop([1.0, 0.8, 0.7, 0.72, 0.69, 0.71, 0.73, 0.74, 0.6], patience=3), (4, 7))


def test_never_stops():
    """Always improving: run to the end"""
    expect_equal(early_stop([5, 4, 3, 2, 1], patience=2), (4, 4))


def test_patience_one():
    """Patience 1 stops at the first non-improvement"""
    expect_equal(early_stop([3, 2, 2.5, 1], patience=1), (1, 2))


def test_min_delta():
    """Improvements smaller than min_delta don't count"""
    expect_equal(early_stop([1.0, 0.999, 0.998, 0.997], patience=2, min_delta=0.01), (0, 2))
    expect_equal(early_stop([1.0, 0.98, 0.979, 0.95], patience=2, min_delta=0.01), (3, 3))


def test_ties():
    """Equal loss is not an improvement"""
    expect_equal(early_stop([1.0, 1.0, 1.0], patience=2), (0, 2))
