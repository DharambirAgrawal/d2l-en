def quick_loss(batch_means):
    """Average of the minibatch means."""
    pass


def exact_loss(batch_means, batch_size, n):
    """Mean loss over all n examples."""
    # Your implementation here
    pass


print(quick_loss([1.0, 1.0, 4.0]), exact_loss([1.0, 1.0, 4.0], 10, 21))
