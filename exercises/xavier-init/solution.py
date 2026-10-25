import numpy as np


def xavier_normal(n_in, n_out, rng):
    """Weights from N(0, 2 / (n_in + n_out))."""
    return rng.normal(0.0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out))


def layer_stds(X, weights, act):
    """Standard deviation of the activations after each layer."""
    H, stds = X, []
    for W in weights:
        H = act(H @ W)
        stds.append(float(H.std()))
    return stds


rng = np.random.default_rng(0)
X = rng.normal(size=(500, 256))
for name, make in [('sigma=0.01', lambda: rng.normal(0, 0.01, (256, 256))),
                   ('sigma=1', lambda: rng.normal(0, 1.0, (256, 256))),
                   ('xavier', lambda: xavier_normal(256, 256, rng))]:
    stds = layer_stds(X, [make() for _ in range(30)], np.tanh)
    if stds is not None:
        print(f'{name:11s}', ' '.join(f'{s:.3f}' for s in stds[::5]))
