import numpy as np


def huber(y_hat, y, sigma=1.0):
    """Mean Huber loss."""
    # Your implementation here
    pass


def huber_grad(y_hat, y, sigma=1.0):
    """Gradient of huber() with respect to y_hat."""
    pass


print(huber(np.array([0.5, 3.0]), np.array([0.0, 0.0])))
