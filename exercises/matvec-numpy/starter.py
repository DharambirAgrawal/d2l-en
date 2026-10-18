import numpy as np


def matvec(A, x):
    """Return the matrix-vector product of A (shape (m, n)) and x (shape (n,))."""
    # Your implementation here (no @, np.dot, np.matmul, ...)
    pass


A = np.array([[1, 2], [3, 4], [5, 6]])
x = np.array([1, 1])
print(matvec(A, x))
