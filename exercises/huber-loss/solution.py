import numpy as np


def huber(y_hat, y, sigma=1.0):
    """Mean Huber loss."""
    r = np.abs(y_hat - y)
    return float(np.mean(np.where(r > sigma, r - sigma / 2, r ** 2 / (2 * sigma))))


def huber_grad(y_hat, y, sigma=1.0):
    """Gradient of huber() with respect to y_hat."""
    d = y_hat - y
    return np.where(np.abs(d) > sigma, np.sign(d), d / sigma) / len(d)


print(huber(np.array([0.5, 3.0]), np.array([0.0, 0.0])))
