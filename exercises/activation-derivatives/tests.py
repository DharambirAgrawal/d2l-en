import numpy as np

from submission import prelu_grad, swish_grad
from grader import expect_close


def numeric(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_prelu():
    """pReLU derivative is 1 on the right and alpha on the left"""
    x = np.array([-3.0, -0.1, 0.1, 3.0])
    expect_close(prelu_grad(x, 0.25), np.array([0.25, 0.25, 1.0, 1.0]))


def test_prelu_numeric():
    """pReLU derivative matches a central difference away from 0"""
    x = np.array([-1.7, -0.3, 0.4, 2.2])
    f = lambda z: np.maximum(0, z) + 0.05 * np.minimum(0, z)
    expect_close(prelu_grad(x, 0.05), numeric(f, x), atol=1e-6)


def test_swish_numeric():
    """Swish derivative matches a central difference for several betas"""
    x = np.linspace(-4, 4, 17)
    for beta in (0.5, 1.0, 2.0):
        f = lambda z: z / (1 + np.exp(-beta * z))
        expect_close(swish_grad(x, beta), numeric(f, x), atol=1e-6)


def test_swish_limits():
    """Swish derivative is 1/2 at 0 and tends to 1 for large x"""
    expect_close(swish_grad(np.array([0.0]), 1.0), np.array([0.5]))
    expect_close(swish_grad(np.array([30.0]), 1.0), np.array([1.0]), atol=1e-9)
