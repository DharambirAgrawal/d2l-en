import numpy as np


def theoretical_variance(p, n):
    """Exact variance of the fraction of heads in n tosses."""
    # Your implementation here
    pass


def simulated_variance(p, n, trials, seed=0):
    """Variance of the fraction of heads across `trials` simulated experiments."""
    # Your implementation here
    pass


for n in (10, 100, 1000):
    print(n, theoretical_variance(0.5, n), simulated_variance(0.5, n, 5000))
