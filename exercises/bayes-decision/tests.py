import numpy as np

from submission import decide
from grader import expect_equal


def test_zero_one():
    """0-1 cost gives the most likely class"""
    rng = np.random.default_rng(0)
    P = rng.random((20, 5)); P /= P.sum(1, keepdims=True)
    C = 1 - np.eye(5)
    expect_equal(list(decide(P, C)), list(P.argmax(1)))


def test_medical():
    """Missing a disease costs 50: flag it at 10% and 3%, not at 1%"""
    P = np.array([[0.9, 0.1], [0.97, 0.03], [0.99, 0.01]])
    C = np.array([[0, 1], [50, 0]])
    expect_equal(list(decide(P, C)), [1, 1, 0])


def test_three_classes():
    """A 'refer to a doctor' option that is never free"""
    # classes: healthy, sick; decisions: healthy, sick, refer (cost 2 always)
    P = np.array([[0.5, 0.5, 0.0], [0.99, 0.01, 0.0]])
    C = np.array([[0, 10, 2], [10, 0, 2], [0, 0, 0]])
    expect_equal(list(decide(P, C)), [2, 0])
