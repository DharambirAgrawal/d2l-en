import numpy as np


def poisson_nll(log_rate, k):
    """Mean Poisson negative log-likelihood (without the log k! constant)."""
    return float(np.mean(np.exp(log_rate) - k * log_rate))


def poisson_nll_grad(log_rate, k):
    """Gradient of poisson_nll with respect to log_rate."""
    return (np.exp(log_rate) - k) / len(k)


print(poisson_nll(np.log(np.array([2.0, 5.0])), np.array([3, 4])))
