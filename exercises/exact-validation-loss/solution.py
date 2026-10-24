def quick_loss(batch_means):
    """Average of the minibatch means."""
    return sum(batch_means) / len(batch_means)


def exact_loss(batch_means, batch_size, n):
    """Mean loss over all n examples."""
    m = len(batch_means)
    last = n - (m - 1) * batch_size
    return (batch_size * sum(batch_means[:-1]) + last * batch_means[-1]) / n


print(quick_loss([1.0, 1.0, 4.0]), exact_loss([1.0, 1.0, 4.0], 10, 21))
