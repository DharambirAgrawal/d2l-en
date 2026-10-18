import numpy as np


def best_squared(x):
    """b minimizing sum((x - b)**2)."""
    return float(np.mean(x))


def best_absolute(x):
    """b minimizing sum(abs(x - b))."""
    return float(np.median(x))


x = np.array([2.0, 3.0, 3.5, 4.0, 1000.0])
print(best_squared(x), best_absolute(x))
