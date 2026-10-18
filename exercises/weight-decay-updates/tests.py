import numpy as np

from submission import l1_step, l2_step
from grader import expect_close, expect_true


def test_l2_is_proportional_shrinkage():
    """With zero data gradient, L2 multiplies w by (1 - lr*lambda)"""
    w = np.array([1.0, -2.0, 0.5])
    expect_close(l2_step(w, np.zeros(3), 0.1, 0.5), w * 0.95)


def test_with_data_gradient():
    """Data gradient and penalty both apply"""
    w, g = np.array([1.0, -1.0]), np.array([0.2, 0.4])
    expect_close(l2_step(w, g, 0.1, 0.5), w - 0.1 * (g + 0.5 * w))
    expect_close(l1_step(w, g, 0.1, 0.5), w - 0.1 * (g + 0.5 * np.sign(w)))


def test_l1_constant_pull():
    """L1 moves every nonzero weight by lr*lambda toward 0"""
    w = np.array([1.0, -1.0, 0.0])
    expect_close(l1_step(w, np.zeros(3), 0.1, 0.5), np.array([0.95, -0.95, 0.0]))


def test_l1_sparsifies_small_weights_faster():
    """After 10 steps, L1 has pushed the small weight much closer to 0"""
    w1 = w2 = np.array([1.0, 0.01])
    for _ in range(10):
        w1 = l1_step(w1, np.zeros(2), 0.001, 1.0)
        w2 = l2_step(w2, np.zeros(2), 0.001, 1.0)
    expect_true(abs(w1[1]) < abs(w2[1]) / 2, 'L1 should drive the small weight toward 0 faster.')
