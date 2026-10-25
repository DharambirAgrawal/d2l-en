import numpy as np


def dropout(X, p, training, rng):
    """Inverted dropout: zero with probability p, scale survivors by 1/(1-p)."""
    if not training or p == 0:
        return X
    if p == 1:
        return np.zeros_like(X)
    mask = rng.random(X.shape) > p
    return mask * X / (1.0 - p)


rng = np.random.default_rng(0)
X = np.arange(16, dtype=float).reshape(2, 8)
print(dropout(X, 0.5, True, rng))
print(dropout(X, 0.5, False, rng))
