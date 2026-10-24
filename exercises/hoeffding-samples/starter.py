import math


def samples_needed(t, delta):
    """Smallest n with 2 * exp(-2 n t^2) <= delta."""
    # Your implementation here
    pass


def error_bar(n, delta):
    """t such that 2 * exp(-2 n t^2) = delta."""
    pass


print(samples_needed(0.01, 0.05), error_bar(10000, 0.001))
