import numpy as np


def cross_entropy(logits, Y):
    """Mean cross-entropy between label distributions Y and softmax(logits)."""
    # Your implementation here
    pass


logits = np.array([[2.0, 1.0, 0.1]])
print(cross_entropy(logits, np.array([[0.9, 0.05, 0.05]])))
