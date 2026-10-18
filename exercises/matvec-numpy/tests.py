import numpy as np

from submission import matvec
from grader import expect_close, expect_raises, expect_shape, expect_true, uses_any


def test_example():
    """[[1, 2], [3, 4], [5, 6]] times [1, 1] = [3, 7, 11]"""
    A = np.array([[1, 2], [3, 4], [5, 6]])
    expect_close(matvec(A, np.array([1, 1])), np.array([3, 7, 11]))


def test_output_shape():
    """A (4, 3) matrix times a (3,) vector has shape (4,)"""
    expect_shape(matvec(np.ones((4, 3)), np.ones(3)), (4,))


def test_random_matrices():
    """Matches A @ x on random float matrices of several shapes"""
    rng = np.random.default_rng(0)
    for m, n in [(1, 1), (2, 5), (5, 2), (16, 16), (30, 7)]:
        A = rng.normal(size=(m, n))
        x = rng.normal(size=n)
        expect_close(matvec(A, x), A @ x, message=f'Wrong result for shape {(m, n)}.')


def test_identity():
    """The identity matrix leaves x unchanged"""
    x = np.array([4.0, -2.0, 0.5])
    expect_close(matvec(np.eye(3), x), x)


def test_mismatched_shapes():
    """Columns of A must match the length of x"""
    expect_raises(ValueError, matvec, np.ones((2, 3)), np.ones(2))


def test_no_builtin_matmul():
    """Doesn't use @, np.dot, np.matmul, np.inner, np.einsum or np.tensordot"""
    used = uses_any('@', 'dot', 'matmul', 'inner', 'einsum', 'tensordot', 'vdot')
    expect_true(not used, 'Your code uses ' + ', '.join(used) + '. Try broadcasting A * x instead.')
