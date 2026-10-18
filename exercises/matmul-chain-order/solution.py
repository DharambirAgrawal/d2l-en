def chain_cost(a, b, c):
    """Compare the cost of (AB)C and A(BC) for matrices with shapes a, b, c."""
    (n, k), (k2, m), (m2, p) = a, b, c
    if k != k2 or m != m2:
        raise ValueError(f'cannot multiply shapes {a}, {b}, {c}')
    ab_c = (n * k * m + n * m * p, n * m)
    a_bc = (k * m * p + n * k * p, k * p)
    return {'(AB)C': ab_c, 'A(BC)': a_bc,
            'best': '(AB)C' if ab_c[0] <= a_bc[0] else 'A(BC)'}


print(chain_cost((2**10, 2**16), (2**16, 2**5), (2**5, 2**14)))
