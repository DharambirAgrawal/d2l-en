import numpy as np


def dropout(X, p, training, rng):
    """Inverted dropout: zero with probability p, scale survivors by 1/(1-p)."""
    # Your implementation here
    pass


rng = np.random.default_rng(0)
X = np.arange(16, dtype=float).reshape(2, 8)
print(dropout(X, 0.5, True, rng))
print(dropout(X, 0.5, False, rng))
