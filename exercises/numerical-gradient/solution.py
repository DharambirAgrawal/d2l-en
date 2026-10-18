import numpy as np


def numerical_gradient(f, x, h=1e-5):
    """Central-difference estimate of the gradient of f at x."""
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = h
        grad[i] = (f(x + step) - f(x - step)) / (2 * h)
    return grad


f = lambda x: 3 * x[0] ** 2 + 5 * np.exp(x[1])
print(numerical_gradient(f, np.array([1.0, 0.0])))   # about [6, 5]
