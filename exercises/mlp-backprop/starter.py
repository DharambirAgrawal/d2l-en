import numpy as np


def forward_backward(X, y, W1, b1, W2, b2, lam):
    """Return (J, grads) for a one-hidden-layer ReLU MLP with
    softmax cross-entropy and L2 on the weights."""
    n = X.shape[0]
    # Forward: Z, H, O, then a stable log-softmax (subtract the row max) and the loss J
    # Backward: dO, dW2, db2, dH, dZ, dW1, db1
    # Your implementation here
    pass


rng = np.random.default_rng(0)
X, y = rng.normal(size=(5, 4)), np.array([0, 2, 1, 2, 0])
W1, b1 = rng.normal(0, 0.5, (4, 6)), np.zeros(6)
W2, b2 = rng.normal(0, 0.5, (6, 3)), np.zeros(3)
J, grads = forward_backward(X, y, W1, b1, W2, b2, lam=0.01)
print(J, {k: v.shape for k, v in grads.items()})
