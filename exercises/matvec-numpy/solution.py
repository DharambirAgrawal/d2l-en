import numpy as np


def matvec(A, x):
    """Return the matrix-vector product of A (shape (m, n)) and x (shape (n,))."""
    A, x = np.asarray(A), np.asarray(x)
    if A.ndim != 2 or x.ndim != 1 or A.shape[1] != x.shape[0]:
        raise ValueError(f'Cannot multiply shapes {A.shape} and {x.shape}')
    # Broadcasting multiplies a_ij * x_j; summing each row gives entry i.
    return (A * x).sum(axis=1)


A = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([1, 1])
print(matvec(A, x))
