import numpy as np


def stream(w, b, num_batches, batch_size, seed, noise=0.01):
    """Yield (X, y) minibatches generated on the fly, reproducibly."""
    # Your implementation here
    yield from ()


for X, y in stream(np.array([2.0, -3.4]), 4.2, 2, 3, seed=0):
    print(X.shape, y.shape)
