import numpy as np


def train(X, y, lr, epochs, batch_size, seed=0):
    """Minibatch SGD for linear regression with hand-derived gradients."""
    rng = np.random.default_rng(seed)
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for start in range(0, len(X), batch_size):
            idx = order[start:start + batch_size]
            residual = X[idx] @ w + b - y[idx]
            w -= lr * X[idx].T @ residual / len(idx)
            b -= lr * residual.mean()
    return w, b


rng = np.random.default_rng(42)
X = rng.standard_normal((1000, 2))
y = X @ np.array([2.0, -3.4]) + 4.2 + 0.01 * rng.standard_normal(1000)
print(train(X, y, lr=0.03, epochs=3, batch_size=32))
