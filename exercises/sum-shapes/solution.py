def reduced_shape(shape, axis, keepdims=False):
    """Shape of X.sum(axis=axis, keepdims=keepdims) for X of the given shape."""
    ndim = len(shape)
    if axis is None:
        axes = set(range(ndim))
    else:
        axes = {a % ndim for a in (axis if isinstance(axis, tuple) else (axis,))}
    out = []
    for i, n in enumerate(shape):
        if i in axes:
            if keepdims:
                out.append(1)
        else:
            out.append(n)
    return tuple(out)


print(reduced_shape((2, 3, 4), 0))
