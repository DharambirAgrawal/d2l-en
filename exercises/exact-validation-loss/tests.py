import numpy as np

from submission import exact_loss, quick_loss
from grader import expect_close


def test_quick():
    """Quick estimate averages the means"""
    expect_close(quick_loss([1.0, 2.0, 6.0]), 3.0)


def test_last_batch_small():
    """Short last batch gets less weight"""
    expect_close(exact_loss([1.0, 1.0, 4.0], 10, 21), 24.0 / 21)


def test_all_full():
    """All batches full: both agree"""
    expect_close(exact_loss([2.0, 4.0], 5, 10), 3.0)
    expect_close(quick_loss([2.0, 4.0]), 3.0)


def test_random():
    """Matches the mean over individual losses"""
    rng = np.random.default_rng(3)
    for n, b in ((1000, 64), (257, 256), (100, 7)):
        losses = rng.exponential(1.0, n)
        means = [losses[i:i + b].mean() for i in range(0, n, b)]
        expect_close(exact_loss(means, b, n), losses.mean())
