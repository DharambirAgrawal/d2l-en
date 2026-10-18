import math


def f_and_grad(x):
    """Return (f(x), f'(x)) for f(x) = log(x^2) * sin(x) + 1/x, via the chain rule."""
    # Forward pass
    a = x * x
    b = math.log(a)
    c = math.sin(x)
    d = b * c
    e = 1 / x
    f = d + e
    # Backward pass: df/d(node), from the output back to x
    df_dd = 1.0
    df_de = 1.0
    df_db = df_dd * c
    df_dc = df_dd * b
    df_da = df_db / a
    df_dx = df_da * 2 * x + df_dc * math.cos(x) + df_de * (-1 / x ** 2)
    return f, df_dx


print(f_and_grad(2.0))
