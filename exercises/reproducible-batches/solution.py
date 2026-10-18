import numpy as np


def stream(w, b, num_batches, batch_size, seed, noise=0.01):
    """Yield (X, y) minibatches generated on the fly, reproducibly."""
    rng = np.random.default_rng(seed)
    for _ in range(num_batches):
        X = rng.standard_normal((batch_size, len(w)))
        y = X @ w + b + noise * rng.standard_normal(batch_size)
        yield X, y


for X, y in stream(np.array([2.0, -3.4]), 4.2, 2, 3, seed=0):
    print(X.shape, y.shape)
