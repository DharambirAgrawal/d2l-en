def tangent_line(f, x0, h=1e-6):
    """Return (slope, intercept) of the tangent line of f at x0."""
    slope = (f(x0 + h) - f(x0 - h)) / (2 * h)
    return slope, f(x0) - slope * x0


f = lambda x: x ** 3 - 1 / x
print(tangent_line(f, 1.0))   # about (4, -4)
