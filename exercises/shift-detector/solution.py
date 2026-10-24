import numpy as np


def detect(Xs, Xt, seed=0):
    """Train a source-vs-target logistic regression; return (held-out accuracy, w, b)."""
    X = np.concatenate([Xs, Xt])
    y = np.concatenate([np.zeros(len(Xs)), np.ones(len(Xt))])
    idx = np.random.default_rng(seed).permutation(len(X))
    X, y = X[idx], y[idx]
    k = int(0.7 * len(X))
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(500):
        p = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= 0.5 * Xtr.T @ (p - ytr) / k
        b -= 0.5 * np.mean(p - ytr)
    acc = np.mean(((Xte @ w + b) > 0) == yte)
    return float(acc), w, float(b)


def importance_weights(Xs, w, b):
    """exp(logit) for each source example, normalized to mean 1."""
    beta = np.exp(Xs @ w + b)
    return beta / beta.mean()


rng = np.random.default_rng(0)
Xs = rng.normal(0, 1, (500, 3))
Xt = rng.normal(0, 1, (500, 3)) + np.array([1.0, 0.0, 0.0])
acc, w, b = detect(Xs, Xt)
print(acc, w)
