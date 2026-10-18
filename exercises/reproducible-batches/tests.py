import numpy as np

from submission import stream
from grader import expect_close, expect_equal, expect_shape, expect_true

W = np.array([2.0, -3.4])


def test_shapes_and_count():
    """Right number of batches and shapes"""
    batches = list(stream(W, 4.2, 5, 8, seed=1))
    expect_equal(len(batches), 5)
    expect_shape(batches[0][0], (8, 2))
    expect_shape(batches[0][1], (8,))


def test_reproducible():
    """Same seed, same data"""
    a = list(stream(W, 4.2, 3, 4, seed=7))
    b = list(stream(W, 4.2, 3, 4, seed=7))
    for (X1, y1), (X2, y2) in zip(a, b):
        expect_close(X1, X2)
        expect_close(y1, y2)


def test_different_seeds_differ():
    """Different seeds give different data"""
    X1 = next(stream(W, 4.2, 1, 4, seed=1))[0]
    X2 = next(stream(W, 4.2, 1, 4, seed=2))[0]
    expect_true(not np.allclose(X1, X2), 'Different seeds should give different data.')


def test_labels_follow_the_model():
    """y is close to X @ w + b"""
    X, y = next(stream(W, 4.2, 1, 500, seed=3))
    expect_close(y, X @ W + 4.2, atol=0.05)


def test_lazy():
    """It's a generator (data made on the fly)"""
    import types
    expect_true(isinstance(stream(W, 4.2, 10**9, 2, seed=0), types.GeneratorType),
                'stream should be a generator (use yield).')
