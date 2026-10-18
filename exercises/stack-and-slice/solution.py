import numpy as np


def stack_and_recover(A, B, C):
    """Stack A, B, C along a new last axis; return (T, B recovered from T)."""
    T = np.stack([A, B, C], axis=-1)
    return T, T[:, :, 1]


rng = np.random.default_rng(0)
A, B, C = (rng.normal(size=(100, 200)) for _ in range(3))
T, B_again = stack_and_recover(A, B, C)
print(T.shape)
