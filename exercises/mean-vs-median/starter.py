import numpy as np


def best_squared(x):
    """b minimizing sum((x - b)**2)."""
    # Your implementation here
    pass


def best_absolute(x):
    """b minimizing sum(abs(x - b))."""
    pass


x = np.array([2.0, 3.0, 3.5, 4.0, 1000.0])
print(best_squared(x), best_absolute(x))
