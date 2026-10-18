import numpy as np


def numerical_gradient(f, x, h=1e-5):
    """Central-difference estimate of the gradient of f at x."""
    # Your implementation here
    pass


f = lambda x: 3 * x[0] ** 2 + 5 * np.exp(x[1])
print(numerical_gradient(f, np.array([1.0, 0.0])))   # about [6, 5]
