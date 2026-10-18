"""d2llite: the parts of the book's `d2l` package used by the early chapters,
for running book cells in the browser (`from d2l import torch as d2l`).

Plotting helpers follow the book's own definitions in
chapter_preliminaries/calculus.md. Anything else raises a clear error.
"""
import sys
import types

import numpy as np
from matplotlib import pyplot as plt


def use_svg_display():
    """Use the svg format to display a plot in Jupyter (no-op in the browser)."""


def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim), axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, 'ndim') and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], '__len__'))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)

    set_figsize(figsize)
    if axes is None:
        axes = plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        x, y = np.asarray(x), np.asarray(y)
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class _D2L(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(
            f'd2l.{name} is not available in the browser yet (this page runs a small '
            'subset of the d2l package). Run this code in Colab or locally.')


def install():
    """Register `d2l`, `d2l.torch` and the IPython display helpers the book imports."""
    me = sys.modules[__name__]
    pkg = _D2L('d2l')
    lib = _D2L('d2l.torch')
    for name in ('use_svg_display', 'set_figsize', 'set_axes', 'plot'):
        setattr(lib, name, getattr(me, name))
    lib.plt = plt
    lib.np = lib.numpy = np
    pkg.torch = lib
    sys.modules['d2l'] = pkg
    sys.modules['d2l.torch'] = lib
    # `from matplotlib_inline import backend_inline` in the book's plotting code
    mi = types.ModuleType('matplotlib_inline')
    bi = types.ModuleType('matplotlib_inline.backend_inline')
    bi.set_matplotlib_formats = lambda *a, **k: None
    mi.backend_inline = bi
    sys.modules.setdefault('matplotlib_inline', mi)
    sys.modules.setdefault('matplotlib_inline.backend_inline', bi)
