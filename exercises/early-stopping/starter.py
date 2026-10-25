def early_stop(val_losses, patience, min_delta=0.0):
    """Return (best_epoch, stop_epoch)."""
    # Your implementation here
    pass


print(early_stop([1.0, 0.8, 0.7, 0.72, 0.69, 0.71, 0.73, 0.74, 0.6], patience=3))
