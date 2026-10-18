import numpy as np


def portfolio_stats(alpha, mu, Sigma):
    """Return (expected return, variance) of a portfolio with weights alpha."""
    # Your implementation here
    pass


mu = np.array([0.05, 0.10, 0.02])
Sigma = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.01]])
print(portfolio_stats(np.array([0.5, 0.3, 0.2]), mu, Sigma))
