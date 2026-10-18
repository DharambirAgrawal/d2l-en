import numpy as np

from submission import poisson_nll, poisson_nll_grad
from grader import expect_close


def test_value():
    """Matches lambda - k log(lambda), averaged"""
    lr = np.log(np.array([2.0, 5.0]))
    k = np.array([3, 4])
    expect_close(poisson_nll(lr, k), np.mean([2 - 3 * np.log(2), 5 - 4 * np.log(5)]))


def test_minimum_at_k():
    """Loss is smallest when lambda equals the count"""
    k = np.array([4.0])
    best = poisson_nll(np.log(k), k)
    for lam in (3.0, 3.9, 4.1, 6.0):
        expect_close(min(best, poisson_nll(np.log([lam]), k)), best)


def test_gradient():
    """Gradient matches a numerical gradient"""
    rng = np.random.default_rng(0)
    lr = rng.normal(size=6)
    k = rng.poisson(3, 6).astype(float)
    num = np.zeros(6)
    for i in range(6):
        e = np.zeros(6); e[i] = 1e-6
        num[i] = (poisson_nll(lr + e, k) - poisson_nll(lr - e, k)) / 2e-6
    expect_close(poisson_nll_grad(lr, k), num, rtol=1e-5, atol=1e-8)
