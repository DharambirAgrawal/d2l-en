import numpy as np


def mlp_forward(X, params):
    """Logits of an MLP: ReLU after every layer except the last."""
    # Your implementation here
    pass


def count_params(sizes):
    """Number of weights and biases of an MLP with these layer sizes."""
    # Your implementation here
    pass


rng = np.random.default_rng(0)
params = [(rng.normal(0, 0.1, (4, 8)), np.zeros(8)), (rng.normal(0, 0.1, (8, 3)), np.zeros(3))]
print(mlp_forward(rng.normal(size=(2, 4)), params))
print(count_params([784, 256, 10]))
