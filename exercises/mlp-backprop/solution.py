import numpy as np


def forward_backward(X, y, W1, b1, W2, b2, lam):
    """Return (J, grads) for a one-hidden-layer ReLU MLP with
    softmax cross-entropy and L2 on the weights."""
    n = X.shape[0]
    Z = X @ W1 + b1
    H = np.maximum(Z, 0)
    O = H @ W2 + b2
    O_shift = O - O.max(axis=1, keepdims=True)
    logP = O_shift - np.log(np.exp(O_shift).sum(axis=1, keepdims=True))
    P = np.exp(logP)
    J = -logP[np.arange(n), y].mean() + lam / 2 * ((W1 ** 2).sum() + (W2 ** 2).sum())
    dO = P.copy()
    dO[np.arange(n), y] -= 1
    dO /= n
    dW2 = H.T @ dO + lam * W2
    db2 = dO.sum(axis=0)
    dH = dO @ W2.T
    dZ = dH * (Z > 0)
    dW1 = X.T @ dZ + lam * W1
    db1 = dZ.sum(axis=0)
    return J, {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}


rng = np.random.default_rng(0)
X, y = rng.normal(size=(5, 4)), np.array([0, 2, 1, 2, 0])
W1, b1 = rng.normal(0, 0.5, (4, 6)), np.zeros(6)
W2, b2 = rng.normal(0, 0.5, (6, 3)), np.zeros(3)
J, grads = forward_backward(X, y, W1, b1, W2, b2, lam=0.01)
print(J, {k: v.shape for k, v in grads.items()})
