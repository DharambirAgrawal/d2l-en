import numpy as np


def softmax(X):
    """Row-wise softmax of a 2-D array, stable for any finite input."""
    # Your implementation here
    pass


print(softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 1000.0, 0.0]])))
