from submission import broadcast_shape
from grader import expect_equal, expect_raises


def test_example():
    """(3, 1) and (1, 2) give (3, 2)"""
    expect_equal(broadcast_shape((3, 1), (1, 2)), (3, 2))


def test_different_number_of_axes():
    """(2, 3, 4) and (3, 1) give (2, 3, 4)"""
    expect_equal(broadcast_shape((2, 3, 4), (3, 1)), (2, 3, 4))


def test_scalar_shape():
    """A scalar shape () broadcasts with anything"""
    expect_equal(broadcast_shape((5,), ()), (5,))
    expect_equal(broadcast_shape((), (2, 2)), (2, 2))


def test_three_dimensional():
    """3-D tensors, as in the section's exercise"""
    expect_equal(broadcast_shape((2, 1, 4), (1, 3, 1)), (2, 3, 4))


def test_incompatible():
    """Mismatched lengths that are not 1 raise ValueError"""
    expect_raises(ValueError, broadcast_shape, (2, 3), (3, 2))
    expect_raises(ValueError, broadcast_shape, (4,), (3,))


def test_matches_numpy():
    """Agrees with np.broadcast_shapes on many random shapes"""
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = tuple(int(v) for v in rng.choice([1, 2, 3], size=rng.integers(0, 4)))
        b = tuple(int(v) for v in rng.choice([1, 2, 3], size=rng.integers(0, 4)))
        try:
            expected = np.broadcast_shapes(a, b)
        except ValueError:
            expect_raises(ValueError, broadcast_shape, a, b)
            continue
        expect_equal(broadcast_shape(a, b), tuple(expected), f'Wrong result for {a} and {b}.')
