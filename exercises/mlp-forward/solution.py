import numpy as np


def mlp_forward(X, params):
    """Logits of an MLP: ReLU after every layer except the last."""
    H = X
    for i, (W, b) in enumerate(params):
        H = H @ W + b
        if i < len(params) - 1:
            H = np.maximum(H, 0)
    return H


def count_params(sizes):
    """Number of weights and biases of an MLP with these layer sizes."""
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


rng = np.random.default_rng(0)
params = [(rng.normal(0, 0.1, (4, 8)), np.zeros(8)), (rng.normal(0, 0.1, (8, 3)), np.zeros(3))]
print(mlp_forward(rng.normal(size=(2, 4)), params))
print(count_params([784, 256, 10]))
