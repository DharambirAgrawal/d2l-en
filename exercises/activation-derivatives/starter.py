import numpy as np


def prelu_grad(x, alpha):
    """Derivative of max(0, x) + alpha * min(0, x), elementwise."""
    # Your implementation here
    pass


def swish_grad(x, beta):
    """Derivative of x * sigmoid(beta * x), elementwise."""
    # Your implementation here
    pass


x = np.array([-2.0, -0.5, 0.5, 2.0])
print(prelu_grad(x, 0.1))
print(swish_grad(x, 1.0))
