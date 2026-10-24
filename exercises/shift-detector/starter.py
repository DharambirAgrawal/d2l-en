import numpy as np


def detect(Xs, Xt, seed=0):
    """Train a source-vs-target logistic regression; return (held-out accuracy, w, b)."""
    # Your implementation here
    pass


def importance_weights(Xs, w, b):
    """exp(logit) for each source example, normalized to mean 1."""
    pass


rng = np.random.default_rng(0)
Xs = rng.normal(0, 1, (500, 3))
Xt = rng.normal(0, 1, (500, 3)) + np.array([1.0, 0.0, 0.0])
acc, w, b = detect(Xs, Xt)
print(acc, w)
