def vector_add(a, b):
    """Return the element-wise sum of vectors a and b (lists of numbers)."""
    if len(a) != len(b):
        raise ValueError(f'Vectors must have the same length, got {len(a)} and {len(b)}')
    return [x + y for x, y in zip(a, b)]


print(vector_add([1, 2], [3, 4]))
