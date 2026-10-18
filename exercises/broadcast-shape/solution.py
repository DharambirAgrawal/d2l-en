def broadcast_shape(a, b):
    """Return the broadcast shape of shapes a and b, or raise ValueError."""
    n = max(len(a), len(b))
    a = (1,) * (n - len(a)) + tuple(a)
    b = (1,) * (n - len(b)) + tuple(b)
    result = []
    for x, y in zip(a, b):
        if x != y and x != 1 and y != 1:
            raise ValueError(f'shapes {a} and {b} cannot be broadcast')
        result.append(max(x, y))
    return tuple(result)


print(broadcast_shape((3, 1), (1, 2)))
