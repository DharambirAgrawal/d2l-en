import numpy as np


def decide(P, C):
    """Predictions minimizing expected cost. P: (n, q) probabilities, C: (q, q) costs."""
    # Your implementation here
    pass


P = np.array([[0.9, 0.1], [0.97, 0.03]])
C = np.array([[0, 1], [50, 0]])  # truth 1 predicted as 0 costs 50
print(decide(P, C))
