def early_stop(val_losses, patience, min_delta=0.0):
    """Return (best_epoch, stop_epoch)."""
    best, best_epoch = float('inf'), 0
    for epoch, loss in enumerate(val_losses):
        if loss < best - min_delta:
            best, best_epoch = loss, epoch
        elif epoch - best_epoch >= patience:
            return best_epoch, epoch
    return best_epoch, len(val_losses) - 1


print(early_stop([1.0, 0.8, 0.7, 0.72, 0.69, 0.71, 0.73, 0.74, 0.6], patience=3))
