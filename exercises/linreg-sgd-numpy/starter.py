import numpy as np


def train(X, y, lr, epochs, batch_size, seed=0):
    """Minibatch SGD for linear regression with hand-derived gradients."""
    # Your implementation here
    pass


rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 2))
y = X @ np.array([2.0, -3.4]) + 4.2 + 0.01 * rng.standard_normal(1000)
print(train(X, y, lr=0.03, epochs=3, batch_size=32))
