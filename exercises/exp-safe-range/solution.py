import math

import numpy as np


def exp_range(dtype):
    """(log of smallest normal, log of largest finite) for a float dtype."""
    info = np.finfo(dtype)
    return math.log(float(info.tiny)), math.log(float(info.max))


def fits(x, dtype):
    """Is exp(x) a normal, finite number of this dtype?"""
    lo, hi = exp_range(dtype)
    return lo <= x <= hi


for t in (np.float16, np.float32, np.float64):
    print(t.__name__, exp_range(t))
