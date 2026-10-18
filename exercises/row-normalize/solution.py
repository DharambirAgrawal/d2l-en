import numpy as np


def row_normalize(A):
    """Return A with each row divided by its sum."""
    A = np.asarray(A, dtype=float)
    return A / A.sum(axis=1, keepdims=True)


A = np.array([[1., 3.], [2., 2.], [0., 5.]])
print(row_normalize(A))
