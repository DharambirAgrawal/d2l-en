import numpy as np

from submission import forward_backward
from grader import expect_close, expect_shape, expect_true


def setup(seed=0, n=7, d=5, h=6, q=4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, q, n)
    P = dict(W1=rng.normal(0, 0.7, (d, h)), b1=rng.normal(0, 0.3, h),
             W2=rng.normal(0, 0.7, (h, q)), b2=rng.normal(0, 0.3, q))
    return X, y, P


def numeric_grad(X, y, P, lam, name, eps=1e-6):
    g = np.zeros_like(P[name])
    for idx in np.ndindex(P[name].shape):
        old = P[name][idx]
        P[name][idx] = old + eps
        up = forward_backward(X, y, P['W1'], P['b1'], P['W2'], P['b2'], lam)[0]
        P[name][idx] = old - eps
        down = forward_backward(X, y, P['W1'], P['b1'], P['W2'], P['b2'], lam)[0]
        P[name][idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g


def test_loss():
    """The loss matches the formula (cross-entropy + L2 on the weights only)"""
    X, y, P = setup()
    Z = np.maximum(X @ P['W1'] + P['b1'], 0) @ P['W2'] + P['b2']
    logp = Z - np.log(np.exp(Z).sum(1, keepdims=True))
    want = -logp[np.arange(len(y)), y].mean() + 0.05 * ((P['W1'] ** 2).sum() + (P['W2'] ** 2).sum())
    J, _ = forward_backward(X, y, P['W1'], P['b1'], P['W2'], P['b2'], 0.1)
    expect_close(J, want)


def test_shapes():
    """Each gradient has its parameter's shape"""
    X, y, P = setup()
    _, g = forward_backward(X, y, P['W1'], P['b1'], P['W2'], P['b2'], 0.0)
    for k in P:
        expect_shape(g[k], P[k].shape, message=f"grads['{k}']")


def test_numeric_no_reg():
    """Gradients match numerical gradients (lambda = 0)"""
    X, y, P = setup(1)
    _, g = forward_backward(X, y, P['W1'], P['b1'], P['W2'], P['b2'], 0.0)
    for k in P:
        expect_close(g[k], numeric_grad(X, y, P, 0.0, k), rtol=1e-4, atol=1e-6, message=f"grads['{k}']")


def test_numeric_reg():
    """Gradients match numerical gradients with weight decay (lambda = 0.3)"""
    X, y, P = setup(2)
    _, g = forward_backward(X, y, P['W1'], P['b1'], P['W2'], P['b2'], 0.3)
    for k in P:
        expect_close(g[k], numeric_grad(X, y, P, 0.3, k), rtol=1e-4, atol=1e-6, message=f"grads['{k}']")


def test_large_logits():
    """Large weights don't overflow (use a stable softmax)"""
    X, y, P = setup(3)
    J, g = forward_backward(X * 100, y, P['W1'] * 10, P['b1'], P['W2'] * 10, P['b2'], 0.0)
    expect_true(np.isfinite(J) and all(np.all(np.isfinite(v)) for v in g.values()), 'loss and gradients are finite')
