import numpy as np


def kfold_indices(n, k):
    """List of k (train_idx, val_idx) pairs covering 0..n-1."""
    # Your implementation here
    pass


def log_rmse(preds, labels):
    """sqrt(mean((log(clip(preds, 1)) - log(labels))^2))."""
    # Your implementation here
    pass


for tr, va in kfold_indices(11, 3) or []:
    print(va, len(tr))
print(log_rmse(np.array([110000.0, 180000.0]), np.array([100000.0, 200000.0])))
