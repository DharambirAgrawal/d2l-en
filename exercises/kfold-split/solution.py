import numpy as np


def kfold_indices(n, k):
    """List of k (train_idx, val_idx) pairs covering 0..n-1."""
    sizes = [n // k + (1 if j < n % k else 0) for j in range(k)]
    out, start = [], 0
    for s in sizes:
        val = np.arange(start, start + s)
        train = np.concatenate([np.arange(0, start), np.arange(start + s, n)])
        out.append((train, val))
        start += s
    return out


def log_rmse(preds, labels):
    """sqrt(mean((log(clip(preds, 1)) - log(labels))^2))."""
    return float(np.sqrt(np.mean((np.log(np.clip(preds, 1, None)) - np.log(labels)) ** 2)))


for tr, va in kfold_indices(11, 3) or []:
    print(va, len(tr))
print(log_rmse(np.array([110000.0, 180000.0]), np.array([100000.0, 200000.0])))
