import numpy as np


def prelu_grad(x, alpha):
    """Derivative of max(0, x) + alpha * min(0, x), elementwise."""
    return np.where(x > 0, 1.0, alpha)


def swish_grad(x, beta):
    """Derivative of x * sigmoid(beta * x), elementwise."""
    s = 1 / (1 + np.exp(-beta * x))
    return s + beta * x * s * (1 - s)


x = np.array([-2.0, -0.5, 0.5, 2.0])
print(prelu_grad(x, 0.1))
print(swish_grad(x, 1.0))
