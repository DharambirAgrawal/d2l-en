import numpy as np


def cross_entropy(logits, Y):
    """Mean cross-entropy between label distributions Y and softmax(logits)."""
    Z = logits - logits.max(axis=1, keepdims=True)
    log_p = Z - np.log(np.exp(Z).sum(axis=1, keepdims=True))
    return float(-(Y * log_p).sum(axis=1).mean())


logits = np.array([[2.0, 1.0, 0.1]])
print(cross_entropy(logits, np.array([[0.9, 0.05, 0.05]])))
