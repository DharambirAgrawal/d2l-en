import numpy as np


def exp_range(dtype):
    """(log of smallest normal, log of largest finite) for a float dtype."""
    # Your implementation here
    pass


def fits(x, dtype):
    """Is exp(x) a normal, finite number of this dtype?"""
    pass


for t in (np.float16, np.float32, np.float64):
    print(t.__name__, exp_range(t))
