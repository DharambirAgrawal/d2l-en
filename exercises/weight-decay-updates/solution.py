import numpy as np


def l2_step(w, grad, lr, lambd):
    """SGD step with L2 penalty (lambda/2)*||w||^2."""
    return w - lr * (grad + lambd * w)


def l1_step(w, grad, lr, lambd):
    """SGD step with L1 penalty lambda*sum|w|."""
    return w - lr * (grad + lambd * np.sign(w))


w = np.array([1.0, 0.01])
print(l2_step(w, np.zeros(2), 0.1, 0.5), l1_step(w, np.zeros(2), 0.1, 0.5))
